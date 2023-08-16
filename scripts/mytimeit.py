import math
import timeit


def _format_time(timespan, precision=3):
    """Formats the timespan in a human readable form"""
    units = ["s", "ms", "\xb5s", "ns"]
    scaling = [1, 1e3, 1e6, 1e9]
    if timespan > 0.0:
        order = min(-int(math.floor(math.log10(timespan)) // 3), 3)
    else:
        order = 3
    scaled_time = timespan * scaling[order]
    unit = units[order]
    return f"{scaled_time:.{precision}g} {unit}"


class TimeitResult(object):
    """
    Object returned by the timeit magic with info about the run.

    Contains the following attributes :

    loops: (int) number of loops done per measurement
    repeat: (int) number of times the measurement has been repeated
    best: (float) best execution time / number
    all_runs: (list of float) execution time of each run (in s)
    compile_time: (float) time of statement compilation (s)
    """

    def __init__(self, loops, repeat, best, worst, all_runs, compile_time, precision):
        self.loops = loops
        self.repeat = repeat
        self.best = best
        self.worst = worst
        self.all_runs = all_runs
        self.compile_time = compile_time
        self._precision = precision
        self.timings = [dt / self.loops for dt in all_runs]

    @property
    def average(self):
        return math.fsum(self.timings) / len(self.timings)

    @property
    def stdev(self):
        mean = self.average
        return (
            math.fsum([(x - mean) ** 2 for x in self.timings]) / len(self.timings)
        ) ** 0.5

    def __str__(self):
        return "{mean} {pm} {std} per loop (mean {pm} std. dev. of {runs} run{run_plural}, {loops} loop{loop_plural} each)".format(
            pm="+-",
            runs=self.repeat,
            loops=self.loops,
            loop_plural="" if self.loops == 1 else "s",
            run_plural="" if self.repeat == 1 else "s",
            mean=_format_time(self.average, self._precision),
            std=_format_time(self.stdev, self._precision),
        )


def nice_timeit(
    stmt="pass",
    setup="pass",
    number=0,
    repeat=None,
    precision=3,
    timer_func=timeit.default_timer,
    globals=None,
):
    """Time execution of a Python statement or expression."""

    if repeat is None:
        repeat = 7 if timeit.default_repeat < 7 else timeit.default_repeat

    timer = timeit.Timer(stmt, setup, timer=timer_func, globals=globals)

    # Get compile time
    compile_time_start = timer_func()
    compile(timer.src, "<timeit>", "exec")
    total_compile_time = timer_func() - compile_time_start

    # This is used to check if there is a huge difference between the
    # best and worst timings.
    # Issue: https://github.com/ipython/ipython/issues/6471
    if number == 0:
        # determine number so that 0.2 <= total time < 2.0
        for index in range(0, 10):
            number = 10**index
            time_number = timer.timeit(number)
            if time_number >= 0.2:
                break

    all_runs = timer.repeat(repeat, number)
    best = min(all_runs) / number
    worst = max(all_runs) / number
    timeit_result = TimeitResult(
        number, repeat, best, worst, all_runs, total_compile_time, precision
    )

    # Check best timing is greater than zero to avoid a
    # ZeroDivisionError.
    # In cases where the slowest timing is lesser than a microsecond
    # we assume that it does not really matter if the fastest
    # timing is 4 times faster than the slowest timing or not.
    if worst > 4 * best and best > 0 and worst > 1e-6:
        print(
            f"The slowest run took {worst / best:.2f} times longer than the "
            f"fastest. This could mean that an intermediate result "
            f"is being cached."
        )

    print(timeit_result)

    if total_compile_time > 0.1:
        print(f"Compiler time: {total_compile_time:.2f} s")
    return timeit_result
