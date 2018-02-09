classdef Dist_MCTS
    properties
        index % which node is it
        time % what time am I at
        my_reward % what am I worth?
        down_branch_reward = 0; % how much are the line of my best children worth
        kids % my  kid nodes
        my_probability = 0 % how likely am I to be selected compared to other kids
        branch_probability = 0; % how likely is my branch to be selected
        depth % how deep in the tree am I?
        max_depth = 20; % how deep in the tree can I go?
        node_status % state of the world at m node
        n_pulls = 0; % how many times have I been pulled?
        min_sampling_threshold = 0.01; % how deep in the tree do I sample?
        alpha = 0.05; % gradient descent rate
        times; % when are my times
        probs; % how strong are my claims
        prob_available = 1.0; % how likely is it that I am available
        raw_reward = 0.0; % how much am I worth unclaimed
        
        use_greedy = false;
        
        use_ucb = false;
        ucb_beta = 0.1*1.41;
        ucb_epsilon = 0.5;
        
        use_ducb = true;
        ducb_cum_pulls = 0;
        ducb_cum_reward = 0;
        ducb_gamma = 0.99;
        ducb_epsilon = 0.5;
        ducb_beta = 0.1*1.41;
        best_child = -1;
        
    end
    methods
        function  b = Dist_MCTS(ii, ti, di, nsi, md, G)
            b.index = ii;
            b.time = ti;
            b.raw_reward = G.nodes{ii}.get_reward(ti);
            b.my_reward = b.raw_reward;
            b.down_branch_reward = b.raw_reward;
            b.depth = di;
            if di == 1
                b.my_probability = 1;
                b.branch_probability = 1;
            end
            b.node_status = nsi;
            b.node_status(ii)= 0;
            b.max_depth = md;
        end
        
        %% plot tree distribution of each task
        function [times, probs] = plot_task_probs(obj, index, times, probs)
            if obj.index == index
                times(end+1) = obj.time;
                probs(end+1) = obj.branch_probability;
                return
            else
                for i=1:length(obj.kids)
                    if obj.kids{i}.branch_probability > obj.min_sampling_threshold
                        [times, probs] = obj.kids{i}.plot_task_probs(index, times, probs);
                    end
                end
            end
        end
        
        %% make kids
        function obj = make_kids(obj, G, coord)
            for i=1:length(G.nodes)
                if obj.node_status(i)
                    m = Dist_MCTS(i, obj.time + G.travel(obj.index, i), obj.depth+1, obj.node_status, obj.max_depth, G);
                    m.prob_available =  (1-coord.get_p_taken(m.index, m.time));
                    dr = m.my_reward - (m.prob_available * m.raw_reward);
                    m.my_reward = m.prob_available * m.raw_reward;
                    m.down_branch_reward = m.down_branch_reward - dr;
                    obj.kids{end+1} = m;
                end
            end
        end
        
        %% UCB
        function [obj, maxI] = ucb(obj)
            obj.n_pulls = obj.n_pulls + 1;
            minR = inf;
            maxR = -inf;
            for i=1:length(obj.kids)                    
                if obj.kids{i}.down_branch_reward < minR
                    minR = obj.kids{i}.down_branch_reward;
                end
                if obj.kids{i}.down_branch_reward > maxR
                    maxR = obj.kids{i}.down_branch_reward;
                end
            end

            if length(obj.kids) > 1
                maxM = -1;
                maxI = -1;
                for i=1:length(obj.kids)
                    rr = (obj.kids{i}.down_branch_reward-minR) / (maxR-minR);
                    ee = obj.kids{i}.ucb_beta*sqrt(obj.kids{i}.ucb_epsilon*log(obj.n_pulls)/max(0.1,obj.kids{i}.n_pulls));
                    if rr + ee > maxM
                        maxM = rr + ee;
                        maxI = i;
                    end
                    obj.kids{i}.n_pulls * 0.9;
                end
            else
                maxI = 1;
            end
        end
        
        
        %% D-UCB
        function [obj, maxI] = ducb(obj)
            obj.n_pulls = obj.n_pulls + 1;
            minR = inf;
            maxR = -inf;
            for i=1:length(obj.kids)                    
                if obj.kids{i}.down_branch_reward < minR
                    minR = obj.kids{i}.down_branch_reward;
                end
                if obj.kids{i}.down_branch_reward > maxR
                    maxR = obj.kids{i}.down_branch_reward;
                end
            end

            if length(obj.kids) > 1
                maxM = -1;
                maxI = -1;
                for i=1:length(obj.kids)
                    obj.kids{i}.ducb_cum_pulls = obj.kids{i}.ducb_gamma * obj.kids{i}.ducb_cum_pulls;
                    obj.kids{i}.ducb_cum_reward = obj.kids{i}.ducb_gamma * obj.kids{i}.ducb_cum_reward;
                    ee = obj.kids{i}.ducb_beta*sqrt(obj.kids{i}.ducb_epsilon*log(obj.ducb_cum_pulls)/max(0.1,obj.kids{i}.ducb_cum_pulls));
                    %rr = obj.kids{i}.ducb_gamma * ((obj.kids{i}.down_branch_reward-minR) / max(0.001,(maxR-minR)) + obj.ducb_cum_reward) / max(0.1,obj.kids{i}.ducb_cum_pulls);
                    rr = (obj.kids{i}.down_branch_reward-minR) / (maxR-minR);
                    if rr + ee > maxM
                        maxM = rr + ee;
                        maxI = i;
                    end
                    
                end 
                if maxI > 0
                    obj.kids{maxI}.ducb_cum_reward = obj.kids{maxI}.ducb_gamma * (obj.kids{maxI}.down_branch_reward-minR) / max(0.001, maxR-minR) + obj.kids{maxI}.ducb_cum_reward;
                    obj.kids{maxI}.ducb_cum_pulls = obj.kids{maxI}.ducb_cum_pulls + obj.kids{maxI}.ducb_gamma;
                end
            else
                maxI = 1;
            end
        end
        
         %% Greedy
        function [obj, maxI] = greedy(obj)
            obj.n_pulls = obj.n_pulls + 1;
            maxI = -1;
            maxR = -inf;
            for i=1:length(obj.kids)                    
                if obj.kids{i}.down_branch_reward > maxR
                    maxR = obj.kids{i}.down_branch_reward;
                    maxI = i;
                end
            end
        end
        
        %% MCTS search tree
        function obj = mcts_search(obj, G, coord)
            
            % update my reward with coord info
            p_rem =  (1-coord.get_p_taken(obj.index, obj.time));
            if p_rem ~= obj.prob_available
                obj.prob_available = p_rem;
                dr = obj.my_reward - (p_rem * obj.raw_reward);
                obj.my_reward = p_rem * obj.raw_reward;
                obj.down_branch_reward = obj.down_branch_reward - dr;
            end
            
            if obj.depth > obj.max_depth
                obj.down_branch_reward = obj.my_reward;
                return
            end
            
            % make kids if I need to
            if isempty(obj.kids)
                obj = obj.make_kids(G, coord);
                if isempty(obj.kids)
                    obj.down_branch_reward = obj.my_reward;
                    return;
                else
                    obj = obj.sample_kids(); % do initial sampling
                end
            end
            
            % select next kid to search
            if obj.use_ucb
                [obj, maxI] = obj.ucb();
            end
            if obj.use_ducb
               [obj, maxI] = obj.ducb();
            end
            if obj.use_greedy
                [obj, maxI] = obj.greedy();
            end
            
            % search selected kid
            if maxI > -1
                obj.kids{maxI} = obj.kids{maxI}.mcts_search(G, coord);
                maxR = -1;
                for i=1:length(obj.kids)
                    if obj.kids{i}.down_branch_reward > maxR
                        maxR = obj.kids{i}.down_branch_reward;
                        obj.best_child = i;
                    end
                    obj.down_branch_reward = maxR  + obj.my_reward;
                end
            end

        end
        
        %% updates probable actions
        function [obj, coord] = sample_tree(obj, coord)
            if obj.depth == 1
                coord = coord.reset();
            end
            coord = coord.add_claim(obj.index, obj.time, obj.branch_probability);
            
            if isempty(obj.kids)
                return
            end
            maxR = -inf;
            maxI = -1;
            for i=1:length(obj.kids)
                if obj.kids{i}.down_branch_reward > maxR
                    maxR = obj.kids{i}.down_branch_reward;
                    maxI = i;
                end
            end
            
            sumPP = 0.0;
            for i=1:length(obj.kids)
                if i == maxI
                    obj.kids{i}.my_probability =  obj.kids{i}.my_probability + obj.alpha*(1 - obj.kids{i}.my_probability);
                else
                    obj.kids{i}.my_probability =  obj.kids{i}.my_probability + obj.alpha*(0 - obj.kids{i}.my_probability);
                end
                sumPP = sumPP+ obj.kids{i}.my_probability;
            end
            
            for i=1:length(obj.kids)
               obj.kids{i}.my_probability = obj.kids{i}.my_probability/sumPP;  % normalize
               obj.kids{i}.branch_probability = obj.branch_probability * obj.kids{i}.my_probability;
               if obj.kids{i}.branch_probability > obj.min_sampling_threshold
                   [obj.kids{i}, coord] = obj.kids{i}.sample_tree(coord);
               end
            end
        end
        
        % creates initial estimate of probable actions
        function obj = sample_kids(obj)
            sumR = 0;
            for i=1:length(obj.kids)
                sumR = sumR + obj.kids{i}.down_branch_reward;
            end
            % minR / maxR does NOT work, puts all 0 to 1 individually,
            % allowing multiple to be 1.0
            for i=1:length(obj.kids)
                obj.kids{i}.my_probability = obj.kids{i}.down_branch_reward/sumR;
                obj.kids{i}.branch_probability = obj.branch_probability * obj.kids{i}.my_probability;
            end
        end
        
        % exploit tree
        function best_path = exploit(obj, best_path)
            best_path(end+1,:) = [obj.index; obj.time; obj.my_reward];
            if obj.depth > obj.max_depth
                return
            end
            
            if isempty(obj.kids)
                return
            else
                maxI = -1;
                maxR = -1;
                for i=1:length(obj.kids)
                    if obj.kids{i}.down_branch_reward > maxR
                        maxR = obj.kids{i}.down_branch_reward;
                        maxI = i;
                    end
                end
                if maxI > -1
                    best_path = obj.kids{maxI}.exploit(best_path);
                else
                    return
                end
            end
        end
    end
    %%%%%%%%%%%%%%%%%% END Dist-MCTS CLASS %%%%%%%%%%%%%%%%%%
end
