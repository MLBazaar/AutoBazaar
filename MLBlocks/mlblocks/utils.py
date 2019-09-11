from datetime import datetime


def log_times(name, append=False):
    def decorator(wrapped):
        def wrapper(self, *args, **kwargs):
            start = datetime.utcnow()
            result = wrapped(self, *args, **kwargs)
            elapsed = (datetime.utcnow() - start).total_seconds()

            if append:
                times = getattr(self, name, None)
                if times is None:
                    times = list()
                    setattr(self, name, times)

                times.append(elapsed)

            else:
                setattr(self, name, elapsed)

            return result

        return wrapper

    return decorator
