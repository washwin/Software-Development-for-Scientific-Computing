#include <stdio.h>
#include <assert.h>

#include "timeutil.h"

void
get_time_now(struct timespec *ts)
{
        if (clock_gettime(CLOCK_MONOTONIC, ts) != 0) {
                perror("clock_gettime failed");
                abort();
        }
}

double
get_time_diff(struct timespec *ts_start, struct timespec *ts_stop)
{
        return ts_stop->tv_sec - ts_start->tv_sec +
                (ts_stop->tv_nsec - ts_start->tv_nsec) * 1E-9;
}
