clc; clear; close;

m0 = [2; 3];
m1 = [6; 5];

s0 = [1 1; 1 2];
s1 = [4 0; 0 1];

s = 0.5 * (s0 + s1);

a = inv(s) * (m1 - m0);

b = (m0 - m1)' * inv(s) * ((m0 + m1)/ 2);

c = 0.5;

part_a = (a' * m0 + b) / (sqrt(a' * s0 * a));

part_b = (a' * m1 + b) / (sqrt(a' * s1 * a));

error = (1 - c) * normcdf(part_a) + c * normcdf(-part_b);