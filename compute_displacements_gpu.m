load('xPhys.mat')
load('f.mat')
load('dimensions.mat')

% f = zeros(6,2);
% f(2,1) = 1;
% f(6,2) = 1;

if ~exist('jK', 'var')
    load('iK.mat')
    load('jK.mat')
    load('Ke.mat')
    load('T_r.mat')

    Ke = tril(Ke) - 0.5*diag(diag(Ke));
    indices = (abs(Ke(:))>0)*ones(1,length(xPhys));
    indices = logical(indices(:)');
    iK = iK(indices);
    jK = jK(indices);
    Ke = Ke(:);
    Ke = Ke(abs(Ke)>0);

    Ke = gpuArray(Ke);
    iK = gpuArray(iK);
    jK = gpuArray(jK);
    T_r = gpuArray(T_r);
    
    iK = iK + 7;
    jK = jK + 7;
    iK = int32(iK);
    jK = int32(jK);
end

ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1) + 6;

sK = (Ke(:) * (xPhys.^double(penal)));
sK = sK(:);

if ndof>210000
    limitdof = 24 * 24 * 16000;
    K = sparse(ndof,ndof);
    loops = round(length(sK)/limitdof);
    for i = 1: loops-1
        K = K + sparse(iK(1+limitdof*(i-1):limitdof*i), jK(1+limitdof*(i-1):limitdof*i), sK(1+limitdof*(i-1):limitdof*i), ndof, ndof);
    end
    i = i+1;
    K = K + sparse(iK(1+limitdof*(i-1):end), jK(1+limitdof*(i-1):end), sK(1+limitdof*(i-1):end), ndof, ndof);
else
    K = sparse(iK, jK, sK, ndof, ndof);
end
% clear sK;

K = T_r' * K * T_r;

nreduced = size(T_r,2);

dofs = 1:ndof;
fixed = 1:3*(nely+1)*(nelz+1);
for i = 0:nelx-1
    fixed2 = 3 + 3 * i * (nelz + 1) * (nely + 1):3:2 + 3 * (nely + 1) + 3 * i * (nelz + 1) * (nely + 1);
    fixed = union(fixed,fixed2);
end
fixed = fixed + 6;
fixed = union(fixed,4);

[~,col,~] = find(K);
x = diff(col);
x = [0;x];
z = col(x>0);
z = [col(1);z];
w = setdiff([1:nreduced]', z);
w = union(w,fixed);
nfixed = length(w);

% a = diag(K);
% a = full(abs(a))>0;
% w = [1:nreduced]';
% w(a) = [];
% w = union(w,fixed);
% nfixed = length(w);

row2 = gpuArray(1:nreduced-nfixed);
col2 = gpuArray(1:nreduced);
col2(w) = [];
v2 = gpuArray(ones(nreduced-nfixed,1));
T_r2 = sparse(row2, col2, v2, nreduced- nfixed, nreduced);
K = T_r2 * K * T_r2';

f(4,:) = [];
% f = [f,[0;0;0;0;1]];%4 gelöscht
F = zeros(length(K),size(f,2));
F(1:5,:) = f * 0.5;

% clear col col2 row row2 v2 z jK iK T_r2;

C = diag(gather(diag(K)))*2;
C = C^-1;
C = sqrt(C);
K = C*K*C;
K = K + K';
F = C*F;
normF = norm(F(:,1));
F = F/normF;

tic
x1 = pcg(K,F,1e-3,1e4);
% x1 = pcg_ohne(K,F,1e-3,1e4);
toc

x1 = C*x1*normF;

w = setdiff([1:nreduced]', w);
x = gpuArray.zeros(nreduced,size(f,2));
x(w,:) = x1;
x = T_r * x;
x = gather(x);

save('x.mat', 'x');
% clear all
% reset(gpuDevice)

function [x] = pcg_ohne(A, b, tol, max_iter)
    % PCG_SOLVER Solves a linear system Ax = b using Preconditioned Conjugate Gradient method.
    % A: The coefficient matrix (square, symmetric, and positive definite).
    % b: The right-hand side vector.
    % tol: The desired tolerance for the relative residual norm (e.g., 1e-6).
    % max_iter: The maximum number of iterations allowed (e.g., 1000).

    x = zeros(size(b)); % Initial guess
    r = b - A * x; % Initial residual
    p = r; % Initial search direction

    j = size(b,2);
    alpha = zeros(1,j);
    beta = zeros(1,j);

    if j == 1
        rr = (r' * r);
        for iter = 1:max_iter
            Ap = A * p;
            alpha = rr / (p' * Ap);
            x = x + alpha * p;
            r = r - alpha * Ap;
            rr_old = rr;
            rr = (r' * r);

            if sqrt(rr) < tol
                break;
            end

            beta= rr / rr_old;
            p = r + beta * p;
        end
    else
        rr = sum(r.*r);
        for iter = 1:max_iter
            Ap = A * p;
            alpha = rr./sum(p.*Ap);
            x = x + alpha .* p;
            r = r - alpha .* Ap;
            rr_old = rr;
            rr = sum(r.*r);

            if sqrt(rr(1,1)) < tol
                break;
            end

            beta = rr ./ rr_old;
            p = r + beta .* p;
        end
    end
end

%     % PCG_SOLVER Solves a linear system Ax = b using Preconditioned Conjugate Gradient method.
%     % A: The coefficient matrix (square, symmetric, and positive definite).
%     % b: The right-hand side vector.
%     % tol: The desired tolerance for the relative residual norm (e.g., 1e-6).
%     % max_iter: The maximum number of iterations allowed (e.g., 1000).
%     % M: The preconditioner (lower triangular matrix obtained from ichol or other methods).
% 
%     x = zeros(size(b)); % Initial guess
%     r = b - A * x; % Initial residual
%     p = r; % Initial search direction
%     n = size(b,2);
%         
%     if n == 1
%         rr = (r' * r);
%         for iter = 1:max_iter
%             Ap = A * p;
%             alpha = rr / (p' * Ap);
%             x = x + alpha * p;
%             r_new = r - alpha * Ap;
%     
%             rr_new = (r_new' * r_new);
%     
%             if rr_new < tol
%                 fprintf('iteration\n')
%                 fprintf('%i\n',iter)
%                 break;
%             end
%            
%             p = r_new + rr_new / rr * p;
%             
%             r = r_new;
%             rr = rr_new;
%         end
%     else
%         rr = (r' * r);
% %         for 1:n
%         for iter = 1:max_iter
%             Ap = p' * A;
%             Ap = Ap';
% 
%             alpha = rr ./ (p' * Ap);
%             alpha = diag(alpha)';
%             x = x + alpha .* p;
%             r_new = r - alpha .* Ap;
%     
%             rr_new = (r_new' * r_new);
%     
%             if rr_new(1,1) < tol
%                 fprintf('iteration\n')
%                 fprintf('%i\n',iter)
%                 break;
%             end
%            
%             p = r_new + diag(rr_new ./ rr)' .* p;
%             
%             r = r_new;
%             rr = rr_new;
%         end
%     end
% end