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