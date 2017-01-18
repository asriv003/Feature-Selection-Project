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