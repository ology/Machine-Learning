#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use_ok 'Data::Science::FromScratch';

my $ds = new_ok 'Data::Science::FromScratch';

is $ds->mean(1,2,3), 6 / 3, 'mean';

is $ds->median(1,10,2,9,5), 5, 'median';
is $ds->median(1,9,2,10), (2 + 9) / 2, 'median';

my @data = (1,1, 2,2,2, 3,3,3,3, 4,4,4,4,4, 5,5,5,5,5,5, 6,6,6,6,6,6);
my @data2 = (4,4, 5,5,5, 6,6,6,6, 7,7,7,7,7, 8,8,8,8,8,8, 9,9,9,9,9,9);

is $ds->quantile(0.10, @data), 2, 'quantile';
is $ds->quantile(0.25, @data), 3, 'quantile';
is $ds->quantile(0.50, @data), 4, 'quantile';
is $ds->quantile(0.75, @data), 5, 'quantile';
is $ds->quantile(0.90, @data), 6, 'quantile';

is_deeply $ds->mode(@data), [5,6], 'mode';

is $ds->data_range(@data), 5, 'data_range';

is_deeply $ds->de_mean(1,2,3), [-1,0,1], 'de_mean';

my $got = $ds->variance(@data);
is sprintf('%.4f', $got), '2.5538', 'variance';

$got = $ds->standard_deviation(@data);
is sprintf('%.4f', $got), '1.5981', 'standard_deviation';

is $ds->interquartile_range(@data), 2, 'interquartile_range';

$got = $ds->covariance(\@data, \@data2);
is sprintf('%.4f', $got), '2.5538', 'covariance';

is $ds->correlation(\@data, \@data2), 1, 'correlation';

# num_friends from the book
@data = (100,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
# daily_minutes from the book
@data2 = (1,68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84);

is $ds->quantile(0.10, @data), 1, 'quantile';
is $ds->quantile(0.25, @data), 3, 'quantile';
is $ds->quantile(0.50, @data), 6, 'quantile';
is $ds->quantile(0.75, @data), 9, 'quantile';
is $ds->quantile(0.90, @data), 13, 'quantile';

is_deeply $ds->mode(@data), [1,6], 'mode';

is $ds->data_range(@data), 99, 'data_range';

$got = $ds->variance(@data);
is sprintf('%.4f', $got), '81.5435', 'variance';

$got = $ds->standard_deviation(@data);
is sprintf('%.4f', $got), '9.0301', 'standard_deviation';

is $ds->interquartile_range(@data), 6, 'interquartile_range';

$got = $ds->covariance(\@data, \@data2);
is sprintf('%.4f', $got), '22.4254', 'covariance';

$got = $ds->correlation(\@data, \@data2);
is sprintf('%.4f', $got), '0.2474', 'correlation';

done_testing();