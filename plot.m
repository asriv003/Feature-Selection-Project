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