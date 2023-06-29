function [x1] = example(problem_case)
    switch(problem_case)
        case 0
            load('K_y.mat');
        case 1
            load('K_z.mat');
        case 2
            load('K.mat');
    end


    load('F.mat');
    K = (K + K')/2;
    q = size(F);
    if q(1) < q(2)
        F = F';
    end

    C = diag(diag(K));
    K_gpu = gpuArray(K);

    try
        [x1,~,~,~,~] = pcg(K_gpu,F,1e-3,1e4,C);
        x1 = gather(x1);
        fprintf('works')
    catch
        fprintf('failed')
        alpha = max(sum(abs(K),2)./diag(K))-2;
        L1 = ichol(K, struct('type','ict','droptol',1e-3,'diagcomp',alpha));
        [x1,~,~,~,~] = pcg(K,F,1e-3,1e4,L1,L1');
    end
end
