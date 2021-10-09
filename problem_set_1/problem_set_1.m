clc; clear; close;

m0 = 3;
s0 = 1;

m1 = 4;
s1 = 3;

x0 = m0 - 3 * s0 : .1 : m0 + 3 * s0;
x1 = m1 - 3 * s1 : .1 : m1 + 3 * s1;

y0 = normpdf(x0, m0, s0);

y1 = normpdf(x1, m1, s1);

figure(1);
plot(x0, y0, "LineWidth", 2)
hold on;
plot(x1, y1, "LineWid", 2)
hold on;

syms x
S = solve(exp(-((x-3)^(2))/(2)) == (exp(-((x-4)^(2))/(18)))/(3), x);

y = 0 : .01 : 0.4;

plot(S(1), y, "k.");
hold on;

plot(S(2), y, "k.");

legend("Y = 0", "Y = 1");

firstPoint = 1.2587;
secondPoint = 4.4913;

Fun0 = @(x) 1/(sqrt(2*pi)*s0)*exp(-(x-m0).^2./(2*s0^2));
Fun1 = @(x) 1/(sqrt(2*pi)*s1)*exp(-(x-m1).^2./(2*s1^2));

error_0 = integral(Fun0, -Inf, firstPoint) + integral(Fun0, secondPoint, Inf);

error_1 = integral(Fun1, firstPoint, secondPoint);

specificity = 1 - error_0;
sensitivity = 1 - error_1;

total_error = error_0 * 0.5 + error_1 * 0.5;