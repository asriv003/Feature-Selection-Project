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
