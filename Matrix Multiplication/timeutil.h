#ifndef TIMEUTIL_H
#define TIMEUTIL_H

#include <stdlib.h>
#include <time.h>

void get_time_now(struct timespec *ts);

double get_time_diff(struct timespec *ts_start, struct timespec *ts_stop);


#endif
