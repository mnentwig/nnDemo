% Neural network demo
% Markus Nentwig, 2018
% largely based on the equations in early chapters of http://neuralnetworksanddeeplearning.com
% Using the same dataset for training and final evaluation(!), 99% hit probability can be achieved.
function neural()
    dataset = load('nist.mat');
    
    nNodes = [400, 30, 10]; % input layer / hidden layer / output layer
    n = numel(nNodes);
    w = {}; % weight
    b = {}; % bias       
    eta = 4; % learning rate
    
    % === set up random starting values ===
    initScale = 1/sqrt(nNodes(1)); % init weight likely not to overdrive the input neurons
    for ix = 2 : n
        w{ix} = randn(nNodes(ix), nNodes(ix-1)) * initScale;
        b(ix) = randn(nNodes(ix), 1);
    end
    
    % === iterate over learning epochs ===
    for ixEpoch = 1:50000
        nBatch = 10;
        
        % === set up average 
        wStep = {}; wStep(2:n) = 0; % note: becomes vector later
        bStep = {}; bStep(2:n) = 0;
        
        % === pick batch ===
        mix = randperm(numel(dataset.y))(1:nBatch);
        for ixSample = mix
            x = [dataset.x(:, ixSample)];
            y = zeros(nNodes(n), 1); 
            y(dataset.y(ixSample)) = 1;
            a = {x};
            ds = {};
            z = {};

            % === forwards calculation ===
            for ix = 2 : n
                z(ix) = w{ix}*a{ix-1}+b{ix};
                [a(ix), ds{ix}] = sigmoid(z{ix});        
            end
            
            % === backwards calculation ===
            for ix = n:-1:2            
                if (ix == n)
                    % === calculate output error ===
                    err{ix} = (a{ix}-y) .* ds{ix};
                else
                    % === calculate error backwards ===
                    err(ix) = (w{ix+1}.' * err{ix+1}) .* ds{ix};
                end
                
                % === average next coefficient update ('stochastic gradient') ===
                wStep{ix} = wStep{ix} + err{ix} * (a{ix-1}.');
                bStep{ix} = bStep{ix} + err{ix};
            end
        end % for batch
        
        % === update with average correction of this batch ===
        for ix = n:-1:2            
            w{ix} = w{ix} - eta * wStep{ix} / nBatch;
            b{ix} = b{ix} - eta * bStep{ix} / nBatch;
        end
 
        % === evaluate network now and then ===
        if mod(ixEpoch, 1000) == 0 
            r = check(dataset, w, b);
            printf('%8i\t%1.3f %%\n', ixEpoch, r*100);
        end
    end
end

function hitRatio = check(dataset, w, b)
    a = {dataset.x};
    z = {};
    n = numel(b);
    for ix = (2 : n)
        z(ix) = w{ix}*a{ix-1}+b{ix}; % note: broadcasting on b in horizontal dimension
        a(ix) = sigmoid(z{ix});        
    end
    
    % === check ===
    [dummy, classification] = max(a{n});
    hitRatio = mean(dataset.y == classification);
end

function [g, dg] = sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z)); % function value
    dg = g .* (1 - g); % derivative
end
