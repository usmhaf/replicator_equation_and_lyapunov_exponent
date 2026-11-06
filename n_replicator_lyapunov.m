% replicator_nstrategies_lyapunov_simple.m
% Solves the replicator ODE for arbitrary number of strategies (n >= 2)
% and computes the finite-time Lyapunov exponent using a single
% (unnormalized) tangent vector, following the RPS example.
%
% NOTE: This "simple" approach is numerically safe when the largest
% Lyapunov exponent is <= 0 (e.g. conservative games such as the
% cyclic RPS-type games). For a chaotic payoff matrix the norm of the
% tangent vector grows exponentially and may overflow for long tspan.
% In that case use the chunked QR version instead.

clear; close all; clc;

%% ----- User inputs (edit as needed) -----
n = 6;

% Payoff matrix (row player).
% Default: cyclic dominance game of length n.
A = zeros(n,n);
for i = 1:n
    j_beat = mod(i, n) + 1;          % i beats j_beat
    j_lose = mod(i-2, n) + 1;        % i loses to j_lose
    A(i, j_beat) = 1;
    A(i, j_lose) = -1;
end

% Initial frequencies (must sum to 1)
x0 = ones(n,1)/n;
x0 = x0 + 0.1*randn(n,1);
x0 = abs(x0);
x0 = x0 / sum(x0);

% Time span for integration
tspan = [0, 1000];
% ----------------------------------------

%% ----- Augmented initial state [x0; v0] -----
% Single random tangent vector, projected onto the simplex tangent
% space {v : sum(v) = 0} and normalized to unit length.
v0 = randn(n,1);
v0 = v0 - mean(v0);           % enforce sum(v0) = 0 (tangent to simplex)
v0 = v0 / norm(v0);           % normalize to unit length

z0 = [x0; v0];                % combined initial state

%% ----- Solve the augmented ODE in one shot -----
odefun_augmented = @(t, z) replicator_with_tangent(z, A, n);
options = odeset('RelTol', 1e-9, 'AbsTol', 1e-12);
[t, z] = ode45(odefun_augmented, tspan, z0, options);

x = z(:, 1:n);                % strategy frequencies
v = z(:, n+1:end);            % tangent vector along trajectory

%% ----- Compute finite-time Lyapunov exponent -----
%   lambda(t) = (1/t) * ln( ||v(t)|| / ||v(0)|| ), with ||v(0)|| = 1
norm_v = sqrt(sum(v.^2, 2));  % Euclidean norm of v at each time
lambda = log(norm_v) ./ t;    % finite-time exponent
lambda(1) = NaN;              % avoid division by zero at t = 0

fprintf('\nFinal finite-time Lyapunov exponent: %+ .6f\n', lambda(end));

%% ----- Plot 0: Lyapunov exponent vs. time -----
figure;
semilogx(t(2:end), lambda(2:end), 'b-', 'LineWidth', 1.5); hold on;
plot([t(2) t(end)], [0 0], 'k--', 'LineWidth', 1);   % zero reference
xlabel('Time (log scale)'); ylabel('\lambda(t)');
title(sprintf('Finite-time Lyapunov exponent – n = %d', n));
grid on;
legend('\lambda(t)', 'zero reference', 'Location', 'best');
ylim([-0.5, 0.5]);           % adjust as needed

%% original plotting code =====

%% n=4 square plot
if n == 4
    figure;
    square_x = [1 1 -1 -1 1];
    square_y = [1 -1 -1 1 1];
    plot(square_x, square_y, 'k-', 'LineWidth', 1.5); hold on;
    axis equal; grid on;
    xlabel('u = (x_1+x_2)-(x_3+x_4)');
    ylabel('v = (x_1+x_3)-(x_2+x_4)');
    title('Trajectory projected onto square');

    u = (x(:,1) + x(:,2)) - (x(:,3) + x(:,4));
    v2 = (x(:,1) + x(:,3)) - (x(:,2) + x(:,4));

    plot(u, v2, 'm-', 'LineWidth', 1.5);
    plot(u(1), v2(1), 'ko', 'MarkerFaceColor','k', 'MarkerSize', 8);
    plot(u(end), v2(end), 'ks', 'MarkerFaceColor','k', 'MarkerSize', 10);

    text(1.05,  1.05, 'Strategy 1');
    text(1.05, -1.05, 'Strategy 2');
    text(-1.05,-1.05, 'Strategy 3');
    text(-1.05, 1.05, 'Strategy 4');
end

%% Plot 1: Time series of frequencies
figure;
plot(t, x, 'LineWidth', 1.5);
xlabel('Time'); ylabel('Frequency');
title('Replicator Dynamics – Strategy Frequencies');
legend(arrayfun(@(i) sprintf('Strategy %d', i), 1:n, 'UniformOutput', false), ...
       'Location', 'best');
grid on; ylim([0 1]);

%% Plot 2: Trajectory in barycentric coordinates
if n == 3
    figure;
    vertices = [0, 1, 0.5; 0, 0, sqrt(3)/2];
    plot([vertices(1,1) vertices(1,2) vertices(1,3) vertices(1,1)], ...
         [vertices(2,1) vertices(2,2) vertices(2,3) vertices(2,1)], ...
         'k-', 'LineWidth', 1.5);
    hold on; axis equal; axis off;
    cart = [x(:,2) + 0.5*x(:,3), sqrt(3)/2 * x(:,3)];
    plot(cart(:,1), cart(:,2), 'm-', 'LineWidth', 1.5);
    plot(cart(1,1), cart(1,2), 'ko', 'MarkerFaceColor','k', 'MarkerSize', 8);
    plot(cart(end,1), cart(end,2), 'ks', 'MarkerFaceColor','k', 'MarkerSize', 10);
    title('Trajectory on 2D Simplex (Triangle)');
    legend('Simplex', 'Trajectory', 'Start', 'End', 'Location', 'best');

elseif n == 4
    figure;
    verts = [0 0 0; 1 0 0; 0 1 0; 0 0 1];
    edges = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4];
    for e = 1:size(edges,1)
        plot3(verts(edges(e,:),1), verts(edges(e,:),2), verts(edges(e,:),3), ...
              'k-', 'LineWidth', 1.5);
        hold on;
    end
    axis equal; grid on; view(3);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    cart = x(:,2:4);
    plot3(cart(:,1), cart(:,2), cart(:,3), 'm-', 'LineWidth', 1.5);
    plot3(cart(1,1), cart(1,2), cart(1,3), 'ko', 'MarkerFaceColor','k', 'MarkerSize', 8);
    plot3(cart(end,1), cart(end,2), cart(end,3), 'ks', 'MarkerFaceColor','k', 'MarkerSize', 10);
    title('Trajectory in 3D Simplex (Tetrahedron)');
    legend('Tetrahedron', 'Trajectory', 'Start', 'End', 'Location', 'best');

else
    warning('For n > 4, plotting projection onto (x_2, x_3, x_4).');
    figure;
    plot3(x(:,2), x(:,3), x(:,4), 'm-', 'LineWidth', 1.5);
    xlabel('x_2'); ylabel('x_3'); zlabel('x_4');
    title('Projection of trajectory onto first three barycentric coordinates');
    grid on; view(3);
end

%% n-gon plot
angles = 2*pi*(0:n-1)/n;
verts  = [cos(angles); sin(angles)];
proj   = (verts * x.').';

figure;
plot([verts(1,:) verts(1,1)], [verts(2,:) verts(2,1)], 'k-', 'LineWidth', 1.5);
hold on; axis equal; grid on;
xlabel('X'); ylabel('Y');
title(sprintf('Trajectory projected onto regular %d-gon', n));
plot(proj(:,1), proj(:,2), 'm-', 'LineWidth', 1.5);
plot(proj(1,1),   proj(1,2),   'ko', 'MarkerFaceColor','k', 'MarkerSize', 8);
plot(proj(end,1), proj(end,2), 'ks', 'MarkerFaceColor','k', 'MarkerSize', 10);

%% ----- Local functions -----
function dzdt = replicator_with_tangent(z, A, n)
    % z = [x; v], where x is n x 1 (simplex) and v is n x 1 (tangent,
    % sum(v) = 0). Computes the nonlinear replicator field and its
    % linearization acting on the single tangent vector v.

    x = z(1:n);
    v = z(n+1:end);

    % 1. Replicator dynamics: dx/dt = x .* (A*x - x'*A*x)
    Ax          = A * x;
    mean_payoff = x.' * Ax;
    dxdt        = x .* (Ax - mean_payoff);

    % 2. Linearised dynamics for tangent vector v: dv/dt = J(x) * v
    %    J(i,j) = delta_ij * (Ax_i - x'Ax) + x_i * (A_ij - (Ax)_j - (A'x)_j)
    Atx = A.' * x;
    J = zeros(n, n);
    for i = 1:n
        for j = 1:n
            J(i,j) = (i==j) * (Ax(i) - mean_payoff) + ...
                     x(i) * (A(i,j) - Ax(j) - Atx(j));
        end
    end
    dvdt = J * v;

    dzdt = [dxdt; dvdt];
end
