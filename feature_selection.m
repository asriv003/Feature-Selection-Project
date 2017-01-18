function feature_selection()
    clc;
    fprintf('Welcome to Abhishek Feature Selection Algorithm\n');
    %filename = input('Type the name of the file to test: ','s');
    %Data = load (filename);
    Data = load ('cs_205_small65.txt', '-ascii');
    Y = Data(:,1);
    X = Data(:,2:end);
    fprintf('\nType the number of the algorithm you want to run\n');
    fprintf('1) Forward Selection\n');
    fprintf('2) Backward Elimination\n');
    fprintf('3) Abhishek Special Algorithm\n');
    ip = input('Input: ');
    
    [m n] = size(X);
    
    fprintf('\n\nThis dataset has %d features (not including the class attribute), with %d instances.\n',n,m);
    fprintf('\n');
    fprintf('Please wait while I normalize the data....');
    %normalize by row
    %X_n = normr(X);
    fprintf('Done!\n\n');
    
    model = fitcknn(X,Y);
    cvmodel = crossval(model,'KFold',m);
    kfloss = kfoldLoss(cvmodel);
    cvacc = 1.00 - kfloss;
    
    fprintf('Running nearest neighbor with add %d features, using "leaving-one-out" evaluation, I get an accuracy of %.1f %%\n\n',n,cvacc*100);
    fprintf('Beginning Search\n\n');
    if ip == 1
        forward_selection(X,Y);   
    elseif ip == 2
        backward_selection(X,Y);
    elseif ip == 3
        abhi_selection(X,Y);
    end
    %forward_selection(X,Y);
    %backward_selection(X,Y);
    %abhi_selection(X,Y);
end