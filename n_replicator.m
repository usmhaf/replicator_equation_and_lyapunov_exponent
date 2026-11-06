% replicator_nstrategies.m
% Solves the replicator ODE for arbitrary number of strategies (n >= 2).
% Plots the trajectory in generalized barycentric coordinates:
%   - n=3: 2D triangle
%   - n=4: 3D tetrahedron
%   - n>4: 3D projection (first three barycentric coordinates after x1)

clear; close all; clc;

%% ----- User inputs (edit as needed) -----
% Number of strategies (should match size of A)
n = 4;

% Payoff matrix (row player). 
% Default: cyclic dominance game of length n.
% Strategy i beats strategy i+1 (mod n), loses to strategy i-1 (mod n),
% and ties with all other strategies.
A = zeros(n,n);
for i = 1:n
    j_beat = mod(i, n) + 1;          % i beats j_beat
    j_lose = mod(i-2, n) + 1;        % i loses to j_lose
    A(i, j_beat) = 1;
    A(i, j_lose) = -1;
end

% Initial frequencies (must sum to 1)
% Start near the interior equilibrium and add a small random perturbation.
x0 = ones(n,1)/n;                    % interior equilibrium (center of simplex)
x0 = x0 + 0.1*randn(n,1);            % perturb randomly
x0 = abs(x0);                        % ensure nonnegative
x0 = x0 / sum(x0);                   % normalize to sum = 1

% Time span for integration
tspan = [0, 50];
% ----------------------------------------

%% Solve replicator ODE
odefun = @(t,x) replicator_rhs(x, A);
options = odeset('RelTol',1e-9,'AbsTol',1e-12);
[t, x] = ode45(odefun, tspan, x0, options);
%% n=4 square plot
% ... after solving the ODE and before the plotting section ...

if n == 4
    % ---------- 2D square projection ----------
    figure;
    % Draw square boundary
    square_x = [1 1 -1 -1 1];
    square_y = [1 -1 -1 1 1];
    plot(square_x, square_y, 'k-', 'LineWidth', 1.5); hold on;
    axis equal; grid on;
    xlabel('u = (x_1+x_2)-(x_3+x_4)');
    ylabel('v = (x_1+x_3)-(x_2+x_4)');
    title('Trajectory projected onto square');

    % Compute projected coordinates
    u = (x(:,1) + x(:,2)) - (x(:,3) + x(:,4));
    v = (x(:,1) + x(:,3)) - (x(:,2) + x(:,4));

    % Plot trajectory
    plot(u, v, 'm-', 'LineWidth', 1.5);
    % Mark start and end
    plot(u(1), v(1), 'ko', 'MarkerFaceColor','k', 'MarkerSize', 8);
    plot(u(end), v(end), 'ks', 'MarkerFaceColor','k', 'MarkerSize', 10);

    % Mark strategy corners (optional)
    text(1.05, 1.05, 'Strategy 1');
    text(1.05, -1.05, 'Strategy 2');
    text(-1.05, -1.05, 'Strategy 3');
    text(-1.05, 1.05, 'Strategy 4');
end


%% Plot 1: Time series of frequencies
figure;
plot(t, x, 'LineWidth', 1.5);
xlabel('Time'); ylabel('Frequency');
title('Replicator Dynamics – Strategy Frequencies');
legend(arrayfun(@(i) sprintf('Strategy %d', i), 1:n, 'UniformOutput', false), ...
       'Location', 'best');
grid on;
ylim([0 1]);

%% Plot 2: Trajectory in barycentric coordinates
if n == 3
    % ---------- 2D simplex (triangle) ----------
    figure;
    % Draw triangle vertices: V1=(0,0), V2=(1,0), V3=(0.5, sqrt(3)/2)
    vertices = [0, 1, 0.5; 0, 0, sqrt(3)/2];
    plot([vertices(1,1) vertices(1,2) vertices(1,3) vertices(1,1)], ...
         [vertices(2,1) vertices(2,2) vertices(2,3) vertices(2,1)], ...
         'k-', 'LineWidth', 1.5);
    hold on; axis equal; axis off;
    % Convert barycentric (x1,x2,x3) to Cartesian:
    % P = x2*V2 + x3*V3  (since V1 is origin)
    cart = [x(:,2) + 0.5*x(:,3), sqrt(3)/2 * x(:,3)];
    plot(cart(:,1), cart(:,2), 'm-', 'LineWidth', 1.5);
    plot(cart(1,1), cart(1,2), 'ko', 'MarkerFaceColor','k', 'MarkerSize', 8);
    plot(cart(end,1), cart(end,2), 'ks', 'MarkerFaceColor','k', 'MarkerSize', 10);
    title('Trajectory on 2D Simplex (Triangle)');
    legend('Simplex', 'Trajectory', 'Start', 'End', 'Location', 'best');
    
elseif n == 4
    % ---------- 3D simplex (tetrahedron) ----------
    figure;
    % Vertices of a right tetrahedron: V1=(0,0,0), V2=(1,0,0), V3=(0,1,0), V4=(0,0,1)
    verts = [0 0 0; 1 0 0; 0 1 0; 0 0 1];
    % All edges of the tetrahedron
    edges = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4];
    for e = 1:size(edges,1)
        plot3(verts(edges(e,:),1), verts(edges(e,:),2), verts(edges(e,:),3), ...
              'k-', 'LineWidth', 1.5);
        hold on;
    end
    axis equal; grid on; view(3);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    % Convert barycentric (x1,x2,x3,x4) to Cartesian:
    % P = x2*V2 + x3*V3 + x4*V4 = (x2, x3, x4)
    cart = x(:,2:4);
    plot3(cart(:,1), cart(:,2), cart(:,3), 'm-', 'LineWidth', 1.5);
    plot3(cart(1,1), cart(1,2), cart(1,3), 'ko', 'MarkerFaceColor','k', 'MarkerSize', 8);
    plot3(cart(end,1), cart(end,2), cart(end,3), 'ks', 'MarkerFaceColor','k', 'MarkerSize', 10);
    title('Trajectory in 3D Simplex (Tetrahedron)');
    legend('Tetrahedron', 'Trajectory', 'Start', 'End', 'Location', 'best');
    
else
    % ---------- For n > 4, show a 3D projection ----------
    warning('For n > 4, the full simplex is (n-1)-dimensional. Plotting projection onto (x_2, x_3, x_4).');
    figure;
    plot3(x(:,2), x(:,3), x(:,4), 'm-', 'LineWidth', 1.5);
    xlabel('x_2'); ylabel('x_3'); zlabel('x_4');
    title('Projection of trajectory onto first three barycentric coordinates');
    grid on; view(3);
end

%% n-gon plot
% Assume you have already solved the ODE and have t and x (m x n matrix)
n = size(x,2);
angles = 2*pi*(0:n-1)/n;               % vertex angles
verts = [cos(angles); sin(angles)];    % 2 x n vertex coordinates

% Project trajectory
proj = (verts * x')';                  % m x 2 matrix

% Plot polygon boundary
figure;
plot([verts(1,:) verts(1,1)], [verts(2,:) verts(2,1)], 'k-', 'LineWidth', 1.5);
hold on; axis equal; grid on;
xlabel('X'); ylabel('Y');
title(sprintf('Trajectory projected onto regular %d-gon', n));

% Plot projected trajectory
plot(proj(:,1), proj(:,2), 'm-', 'LineWidth', 1.5);
plot(proj(1,1), proj(1,2), 'ko', 'MarkerFaceColor','k', 'MarkerSize', 8); % start
plot(proj(end,1), proj(end,2), 'ks', 'MarkerFaceColor','k', 'MarkerSize', 10); % end

%% Local function: replicator equation RHS
function dxdt = replicator_rhs(x, A)
    % x: column vector of frequencies (n x 1)
    % A: payoff matrix (n x n)
    avg_payoff = A * x;               % average payoff of each strategy
    mean_payoff = x' * avg_payoff;    % mean payoff of the population
    dxdt = x .* (avg_payoff - mean_payoff);
end
