clc; clear; close;

m0 = [0; 0];
m1 = [2; 0];

s0 = [2 0; 0 1];
s1 = [2 0; 0 4];

A = 0.5 * (inv(s0) - inv(s1));

b = inv(s1) * m1 - inv(s0) * m0;

c = 0.5 * (m0' * inv(s0) * m0 - m1' * inv(s1) * m1) + 0.5 * log(det(s0) / det(s1));

x = sym("x", [2 1]);
S = solve(x.' * A * x + b.' * x + c == 0, x);

plotErrorEllipse(m0, s0, 0.5)
plotErrorEllipse(m1, s1, 0.5)

xlabel("X1");
ylabel("X0");

f1 = @(x1) - (3*x1^2)/8 - c;
x = [];

idx = 1;
for i = -4 : 0.1 : 4
    x(idx, 2) = i; % X1
    x(idx, 1) = f1(i); % X0
    idx = idx + 1;
end

plot(x(:, 2), x(:, 1), "LineWidth", 2);

% Test
num = 4;
for i = 0 : 10
    x = [-num + (num+num)*rand; -num + (num+num)*rand];
    if x.' * A * x + b.' * x + c > 0
        plot(x(2), x(1), ".r", "MarkerSize", 20);
    else
        plot(x(2), x(1), ".m", "MarkerSize", 20);
    end
end

legend("Y = 0", "mean", "Y = 1", "", "Optimal DB");

function plotErrorEllipse(mu, Sigma, p)

s = -2 * log(1 - p);

[V, D] = eig(Sigma * s);

t = linspace(0, 2 * pi);
a = (V * sqrt(D)) * [cos(t(:))'; sin(t(:))'];

plot(a(1, :) + mu(2), a(2, :) + mu(1), "LineWidth", 2);
hold on;
plot(mu(2), mu(1), "k.", "MarkerSize", 20);
hold on;
end