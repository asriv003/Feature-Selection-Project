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

function forward_selection(X,Y)
    [m n] = size(X);
    test = 0;
    temp = 0;
    bestf = 0;
    bestacc = 0;
    tempacc = 0;
    %selecting first row
    for i = 1:n
        mdl = fitcknn(X(:,i),Y);
        cvmdl = crossval(mdl,'KFold',m);
        kloss = kfoldLoss(cvmdl);
        acc = 1.00 - kloss;
        fprintf('Using feature(s) {%d} accuracy is %.1f %%\n',i,acc*100);
        if tempacc < acc
            bestf = i;
            bestacc = acc;
            tempacc = acc;
        end
    end
    fprintf('Feature set {%d} was best accuracy is %.1f %%\n',bestf,bestacc*100);
    feature = 0;
    temp = bestf;
    while size(temp,2) < n-1
        tempacc = 0;
        if size(temp,2)==1
            test = X(:,temp(1));
        else
            test = [test X(:,temp(size(temp,2)))];
        end
        for i = 1:n
            if ~ismember(i, temp(:))
                mdl = fitcknn([test X(:,i)],Y);
                cvmdl = crossval(mdl,'KFold',m);
                kloss = kfoldLoss(cvmdl);
                acc = 1.00 - kloss;
                g = sprintf('%d ', [temp i]);
                fprintf('Using feature(s) {%s} accuracy is %.1f %%\n',g,acc*100);
                if tempacc < acc
                    feature = i;
                    pg = g;
                    tempacc = acc;
                end
            end
        end
        
        temp = [temp feature];
        if bestacc < tempacc
            bestf = [bestf feature];
            bestacc = tempacc;
        else
            fprintf('Warning, Accuracy has decreased! Continuing search in case of local maxima\n');
        end
        fprintf('Feature set {%s} was best accuracy is %.1f %%\n',pg,tempacc*100);
    end
    bg = sprintf('%d ', bestf);
    fprintf('Finished search!! The best feature subset {%s} which has accuracy of %.1f %%\n',bg,bestacc*100);
end

function backward_selection(X,Y)
    [m n] = size(X);
    temp = 1:n;
    test = X;
    bestf = temp;
    bestacc = 0;
    
    mdl = fitcknn(X,Y);
    cvmdl = crossval(mdl,'KFold',m);
    kloss = kfoldLoss(cvmdl);
    acc = 1.00 - kloss;
    ag = sprintf('%d ', temp);
    fprintf('Using feature(s) {%s} accuracy is %.1f %%\n',ag,acc*100);
    bestacc = acc;
    fprintf('Feature set {%s} was best accuracy is %.1f %%\n',ag,bestacc*100);
    feature = 0;
    while size(temp,2)>1
        tempacc = 0;
        for i = 1:size(temp,2)
            test = horzcat(X(:,1:temp(i)-1),X(:,temp(i)+1:end));
            ttemp = horzcat(temp(:,1:i-1),temp(:,i+1:end));
            mdl = fitcknn(test,Y);
            cvmdl = crossval(mdl,'KFold',m);
            kloss = kfoldLoss(cvmdl);
            acc = 1.00 - kloss;
            bg = sprintf('%d ', ttemp);
            fprintf('Using feature(s) {%s} accuracy is %.1f %%\n',bg,acc*100);
            if tempacc < acc
                feature = i;
                pg = bg;
                tempacc = acc;
            end
        end
        %X = horzcat(X(:,1:temp(feature)-1),X(:,temp(feature)+1:end));
        X(:,temp(feature)) = 0; 
        temp = horzcat(temp(:,1:feature-1),temp(:,feature+1:end));
        if bestacc < tempacc
            bestf = temp;
            bestacc = tempacc;
        else
            fprintf('Warning, Accuracy has decreased! Continuing search in case of local maxima\n');
        end
        fprintf('Feature set {%s} was best accuracy is %.1f %%\n',pg,tempacc*100);
    end
    bg = sprintf('%d ', bestf);
    fprintf('Finished search!! The best feature subset {%s} which has accuracy of %.1f %%\n',bg,bestacc*100);
end

function abhi_selection(X,Y)
    [m n] = size(X);
    ens1 = fitensemble(X,Y,'AdaBoostM1',100,'Tree');
    imp1 = predictorImportance(ens1);
    [sortedValues,sortIndex] = sort(imp1(:),'descend');
    maxIndex = sortIndex(1:5);
    %for i = 1:size(maxIndex,1)
    %    if i==1
    %        test = X(:,maxIndex(i));
    %    else
    %        test = [test X(:,maxIndex(i))];
    %    end
    %end
    %abhi_forward_selection(test,Y,maxIndex);
    acc = 0;
    for i = 1:size(maxIndex,1)
        test = X(:,maxIndex(i));
        mdl = fitcknn( test,Y);
        cvmdl = crossval(mdl,'KFold',m);
        kloss = kfoldLoss(cvmdl);
        acc = [acc 1.00 - kloss];
    end
    [sortedValues2,sortFeature] = sort(imp1(:),'descend');
    maxFeature = sortFeature(1:2);
    sample = [X(:,maxFeature(1)) X(:,maxFeature(2))];

    mdl2 = fitcknn(sample,Y);
    cvmdl2 = crossval(mdl2,'KFold',m);
    kloss2 = kfoldLoss(cvmdl2);
    acc2 = 1.00 - kloss2;
    g = sprintf('%d ', maxFeature);
    fprintf('Using feature(s) {%s} accuracy is %.1f %%\n',g,acc2*100);
end

function abhi_forward_selection(X,Y,maxIndex)
    [m n] = size(X);
    test = 0;
    temp = 0;
    bestf = 0;
    bestacc = 0;
    tempacc = 0;
    %selecting first row
    for i = 1:n
        mdl = fitcknn(X(:,i),Y);
        cvmdl = crossval(mdl,'KFold',m);
        kloss = kfoldLoss(cvmdl);
        acc = 1.00 - kloss;
        fprintf('Using feature(s) {%d} accuracy is %.1f %%\n',maxIndex(i),acc*100);
        if tempacc < acc
            bestf = i;
            bestacc = acc;
            tempacc = acc;
        end
    end
    fprintf('Feature set {%d} was best accuracy is %.1f %%\n',maxIndex(bestf),bestacc*100);
    feature = 0;
    tbestf = maxIndex(bestf);
    temp = bestf;
    ttemp = maxIndex(bestf);
    while size(temp,2) < n-1
        tempacc = 0;
        if size(temp,2)==1
            test = X(:,temp(1));
        else
            test = [test X(:,temp(size(temp,2)))];
        end
        for i = 1:n
            if ~ismember(i, temp(:))
                mdl = fitcknn([test X(:,i)],Y);
                cvmdl = crossval(mdl,'KFold',m);
                kloss = kfoldLoss(cvmdl);
                acc = 1.00 - kloss;
                g = sprintf('%d ', [temp i]);
                tg = sprintf('%d ', [ttemp maxIndex(i)]);
                fprintf('Using feature(s) {%s} accuracy is %.1f %%\n',tg,acc*100);
                if tempacc < acc
                    feature = i;
                    pg = tg;
                    tempacc = acc;
                end
            end
        end
        
        temp = [temp feature];
        ttemp = [ttemp maxIndex(feature)];
        if bestacc < tempacc
            bestf = [bestf feature];
            tbestf = [tbestf maxIndex(feature)];
            bestacc = tempacc;
        else
            fprintf('Warning, Accuracy has decreased! Continuing search in case of local maxima\n');
        end
        fprintf('Feature set {%s} was best accuracy is %.1f %%\n',pg,tempacc*100);
    end
    bg = sprintf('%d ', tbestf);
    fprintf('Finished search!! The best feature subset {%s} which has accuracy of %.1f %%\n',bg,bestacc*100);
end


function plot(X)
    for i = 1:size(X,2)
        for j = i+1:size(X,2)
            d = 9*(i-1)+j-1;
            figure(d);
            gscatter(X(:,i),X(:,j),Y,'br','xo');
            str=sprintf('feature=%d vs feature=%d',i,j);
            title(str)
        end
    end
end
