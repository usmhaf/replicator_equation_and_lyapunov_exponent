% replicator_rps_lyapunov.m
% Solves the replicator ODE for Rock-Paper-Scissors and computes the
% finite-time Lyapunov exponent along the trajectory.

clear; close all; clc;

%% 1. Payoff matrix for RPS (row player)
A = [ 0  -1   1;
      1   0  -1;
     -1   1   0 ];

%% 2. Augmented ODE: state = [x; v], where v is a tangent vector
%   Replicator dynamics preserve sum(x)=1, and v is kept orthogonal to (1,1,1)
odefun_augmented = @(t, xv) rps_with_tangent(xv, A);

%% 3. Initial conditions
x0 = [0.5; 0.3; 0.2];                % strategy frequencies
v0 = randn(3,1);                      % random initial perturbation
v0 = v0 - mean(v0);                   % enforce sum(v0) = 0 (tangent to simplex)
v0 = v0 / norm(v0);                   % normalize to unit length
z0 = [x0; v0];                        % combined initial state

tspan = [0, 50];                      % integrate a bit longer to see asymptotic behaviour

%% 4. Solve the augmented ODE
options = odeset('RelTol',1e-9,'AbsTol',1e-12);
[t, z] = ode45(odefun_augmented, tspan, z0, options);
x = z(:,1:3);   % strategies
v = z(:,4:6);   % perturbation vector

%% 5. Compute finite-time Lyapunov exponent
%   λ(t) = (1/t) * ln( ||v(t)|| / ||v(0)|| ), with ||v(0)|| = 1
norm_v = sqrt(sum(v.^2, 2));          % Euclidean norm of v at each time
lambda = log(norm_v) ./ t;            % finite-time exponent (NaN for t=0)
lambda(1) = NaN;                      % avoid division by zero

%% 6. Plot time series of frequencies
figure('Position', [100, 100, 900, 400]);
subplot(1,2,1);
plot(t, x(:,1), 'r-', 'LineWidth', 1.5); hold on;
plot(t, x(:,2), 'b-', 'LineWidth', 1.5);
plot(t, x(:,3), 'g-', 'LineWidth', 1.5);
xlabel('Time'); ylabel('Frequency');
title('Replicator Dynamics – RPS Game');
legend('Rock', 'Paper', 'Scissors', 'Location', 'best');
grid on; ylim([0 1]);

%% 7. Plot trajectory on the simplex
subplot(1,2,2);
triangle_x = [0, 1, 0.5, 0];
triangle_y = [0, 0, sqrt(3)/2, 0];
plot(triangle_x, triangle_y, 'k-', 'LineWidth', 1.5); hold on;
axis equal; axis off;
text(0, -0.05, 'Rock', 'HorizontalAlignment', 'center');
text(1, -0.05, 'Paper', 'HorizontalAlignment', 'center');
text(0.5, sqrt(3)/2+0.05, 'Scissors', 'HorizontalAlignment', 'center');

x2d = x(:,2) + 0.5 * x(:,3);
y2d = sqrt(3)/2 * x(:,3);
plot(x2d, y2d, 'm-', 'LineWidth', 1.5);
plot(x2d(1), y2d(1), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6);
plot(x2d(end), y2d(end), 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 8);
legend('Simplex', 'Trajectory', 'Start', 'End', 'Location', 'best');
title('Trajectory on the Simplex');

%% 8. Plot Lyapunov exponent vs. time
figure;
semilogx(t(2:end), lambda(2:end), 'b-', 'LineWidth', 1.5); hold on;
yline(0, 'k--', 'LineWidth', 1);      % reference zero line
xlabel('Time (log scale)'); ylabel('\lambda(t)');
title('Finite-time Lyapunov exponent – RPS replicator');
grid on;
legend('\lambda(t)', 'zero reference', 'Location', 'best');
ylim([-0.5, 0.5]);   % adjust as needed

%% Local functions

function dzdt = rps_with_tangent(z, A)
    % z = [x; v], with x in 3-simplex and v tangent (sum v = 0)
    x = z(1:3);
    v = z(4:6);

    % 1. Replicator dynamics: dx/dt = x.*(A*x - x'*A*x)
    Ax = A * x;
    mean_payoff = x' * Ax;
    dxdt = x .* (Ax - mean_payoff);

    % 2. Linearised dynamics for the tangent vector v: dv/dt = J(x) * v
    %    Jacobian J of the replicator vector field f(x)
    %    f_i = x_i * ( (Ax)_i - x'Ax )
    %    df_i/dx_j = delta_ij * ( (Ax)_i - x'Ax ) + x_i * ( A_ij - (Ax)_j - (A'*x)_j )
    Atx = A' * x;           % (A^T x)_j
    J = zeros(3,3);
    for i = 1:3
        for j = 1:3
            delta_ij = (i == j);
            J(i,j) = delta_ij * (Ax(i) - mean_payoff) + ...
                     x(i) * ( A(i,j) - Ax(j) - Atx(j) );
        end
    end
    dvdt = J * v;

    dzdt = [dxdt; dvdt];
end
