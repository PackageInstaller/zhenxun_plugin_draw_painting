"""Resource exhaustion is transient, not an invalid painting or prediction."""


class InferenceResourceError(RuntimeError):
    """The session is cooling down after a failed memory recovery."""


def is_memory_error(error: Exception | str) -> bool:
    if isinstance(error, InferenceResourceError | MemoryError):
        return True
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "bfcarena::allocaterawinternal",
            "out of memory",
            "cudaerrormemoryallocation",
            "cublas_status_alloc_failed",
            "cudnn_status_alloc_failed",
            "failed to allocate memory",
            "bad allocation",
            "bad_alloc",
        )
    )
