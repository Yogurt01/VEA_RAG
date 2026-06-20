import multiprocessing as mp
import queue
import time
import traceback


def _worker_loop(model_path, model_kwargs, task_queue, result_queue, ready_event):
    """
    Chạy bên trong subprocess con.
    Nhận task (task_id, video_path, prompt) từ task_queue,
    trả (task_id, status, payload) qua result_queue.
    status: "ok" | "error"
    """
    # Import bên trong worker để tránh khởi tạo CUDA context ở process cha
    from qwen3_vl_visual import Qwen3VLStandalone

    model = Qwen3VLStandalone(model_path=model_path, **model_kwargs)
    try:
        model.load_model()
    except Exception as e:
        # Báo lỗi load model ngay, để cha không treo chờ ready_event
        result_queue.put(("__load_error__", "error", f"{e}\n{traceback.format_exc()}"))
        return

    ready_event.set()

    while True:
        try:
            task = task_queue.get(timeout=5)
        except queue.Empty:
            continue

        if task is None:  # tín hiệu dừng
            break

        task_id, video_path, prompt = task
        try:
            result = model.predict(video_path=video_path, prompt=prompt, parse_json=True)
            result_queue.put((task_id, "ok", result))
        except Exception as e:
            result_queue.put((task_id, "error", f"{e}\n{traceback.format_exc()}"))
            # Lỗi Python bình thường (không phải native crash) -> worker
            # vẫn sống tiếp, không cần respawn.


class CaptionWorkerPool:
    """
    Quản lý 1 subprocess worker, tự động respawn nếu worker chết
    đột ngột (native crash) trong lúc xử lý 1 task.

    Cách dùng:
        pool = CaptionWorkerPool(model_path, model_kwargs)
        pool.start()
        status, payload = pool.run_task(video_path, prompt, timeout=120)
        # status == "ok"        -> payload là kết quả predict()
        # status == "error"     -> payload là traceback string (lỗi Python thường)
        # status == "crashed"   -> worker chết đột ngột (native crash),
        #                          worker đã được respawn tự động, tasked failed.
        pool.stop()
    """

    def __init__(self, model_path, model_kwargs=None, start_timeout=300):
        self.model_path = model_path
        self.model_kwargs = model_kwargs or {}
        self.start_timeout = start_timeout

        self.ctx = mp.get_context("spawn")
        self.task_queue = None
        self.result_queue = None
        self.ready_event = None
        self.proc = None
        self._task_counter = 0

    def start(self):
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        self.ready_event = self.ctx.Event()

        self.proc = self.ctx.Process(
            target=_worker_loop,
            args=(self.model_path, self.model_kwargs, self.task_queue,
                  self.result_queue, self.ready_event),
            daemon=True,
        )
        self.proc.start()

        ready = self.ready_event.wait(timeout=self.start_timeout)
        if not ready:
            if not self.proc.is_alive():
                raise RuntimeError(
                    f"Caption worker died during startup (exitcode={self.proc.exitcode})"
                )
            raise TimeoutError("Caption worker did not become ready in time")

        # Nếu load model lỗi (Python exception), nó sẽ có message trong queue
        try:
            task_id, status, payload = self.result_queue.get_nowait()
            if task_id == "__load_error__":
                raise RuntimeError(f"Caption worker failed to load model:\n{payload}")
        except queue.Empty:
            pass

    def _respawn(self):
        self._terminate_quiet()
        self.start()

    def _terminate_quiet(self):
        if self.proc is not None and self.proc.is_alive():
            try:
                self.proc.terminate()
                self.proc.join(timeout=10)
            except Exception:
                pass
        self.proc = None

    def run_task(self, video_path, prompt, timeout=180):
        """
        Gửi 1 task tới worker, chờ kết quả tối đa `timeout` giây.
        Trả về (status, payload):
          status: "ok" | "error" | "crashed" | "timeout"
        Nếu status là "crashed", worker đã được respawn tự động trước khi return.
        """
        if self.proc is None or not self.proc.is_alive():
            # Worker đã chết từ trước (vd: crash ở task trước nhưng chưa kịp respawn)
            self._respawn()

        self._task_counter += 1
        task_id = self._task_counter
        self.task_queue.put((task_id, video_path, prompt))

        deadline = time.time() + timeout
        while time.time() < deadline:
            # Worker chết đột ngột giữa chừng -> native crash
            if not self.proc.is_alive():
                dead_proc = self.proc
                dead_proc.join(timeout=2)  # đảm bảo OS đã reap, exitcode chính xác
                exitcode = dead_proc.exitcode
                self._respawn()
                return "crashed", f"Worker process died (exitcode={exitcode}) while processing this clip."

            try:
                result_task_id, status, payload = self.result_queue.get(timeout=1)
            except queue.Empty:
                continue

            if result_task_id != task_id:
                # Kết quả của task cũ (không nên xảy ra vì xử lý tuần tự), bỏ qua
                continue

            return status, payload

        # Hết timeout, coi như worker bị treo -> respawn để không kẹt mãi
        self._respawn()
        return "timeout", f"Worker did not respond within {timeout}s for this clip; worker respawned."

    def stop(self):
        if self.proc is not None and self.proc.is_alive():
            try:
                self.task_queue.put(None)
                self.proc.join(timeout=15)
            except Exception:
                pass
        self._terminate_quiet()