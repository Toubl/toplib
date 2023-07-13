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
%     K = (K + K')/2;
    F = F';

    C = diag(diag(K));
    [x1,~,~,~,~] = pcg(K,F,1e-3,1e4,C);
    fprintf('works')
end
