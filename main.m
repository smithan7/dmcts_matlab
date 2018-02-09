close all
clc

com_mod = [1,5,10,25,50,100,200,400,500,1000];
big_coord_tracker = zeros(length(com_mod),2000);

for big_iter = 1:length(com_mod)
    percent_done = (big_iter-1) / length(com_mod)
    clearvars -except big_coord_tracker big_iter com_mod
    
    rng(1);
    n_nodes = 20;
    G = World(n_nodes);

    index = 1;
    time = 0;
    depth = 1;
    max_depth = 10;
    opt_iters = 2000;

    %%%%%%%%%%%%%%%%%%% Greedy Test %%%%%%%%%%%%%%%%%%
    % greedy_a = Greedy(index, time, depth, node_status, max_depth, G);
    % greedy_a = greedy_a.greedy_search(G);
    % greedy_coord = Coordinator(G.n_nodes);
    % for i=1:opt_iters
    %     [greedy_a, greedy_coord]  = greedy_a.sample_tree(greedy_coord);
    % end
    % 
    % greedy_a.down_branch_reward
    % greedy_path = [];
    % greedy_path = greedy_a.exploit(greedy_path)
    % figure(1)
    %     for i=1:n_nodes
    %         plot(G.nodes{i}.x, G.nodes{i}.y, 'rx')
    %         hold on
    %     end
    %     grid on
    %     for i=1:length(greedy_path)-1
    %         plot([G.nodes{greedy_path(i,1)}.x, G.nodes{greedy_path(i+1,1)}.x], [G.nodes{greedy_path(i,1)}.y, G.nodes{greedy_path(i+1,1)}.y], 'g');
    %     end
    %     plot(G.nodes{greedy_path(1,1)}.x, G.nodes{greedy_path(1,1)}.y, 'go')
    %     plot(G.nodes{greedy_path(end,1)}.x, G.nodes{greedy_path(end,1)}.y, 'go')

    %%%%%%%%%%%%%%%%%% MCTS Test %%%%%%%%%%%%%%%%%%%%%%
    % mcts_a = MCTS(index, time, depth, node_status, max_depth, G);
    % for i=1:opt_iters
    %     mcts_a = mcts_a.mcts_search(G);
    %     mcts_a = mcts_a.sample_tree();
    % end
    % 
    % mcts_a.down_branch_reward
    % mcts_path = [];
    % mcts_path = mcts_a.exploit(mcts_path);
    % figure(2)
    %     for i=1:n_nodes
    %         plot(G.nodes{i}.x, G.nodes{i}.y, 'rx')
    %         hold on
    %     end
    %     grid on
    %     for i=1:length(mcts_path)-1
    %         plot([G.nodes{mcts_path(i)}.x, G.nodes{mcts_path(i+1)}.x], [G.nodes{mcts_path(i)}.y, G.nodes{mcts_path(i+1)}.y], 'g');
    %     end
    %     plot(G.nodes{mcts_path(1)}.x, G.nodes{mcts_path(1)}.y, 'go')
    %     plot(G.nodes{mcts_path(end)}.x, G.nodes{mcts_path(end)}.y, 'go')

    %%%%%%%%%%%%%%%%%% Dist-MCTS Test %%%%%%%%%%%%%%%%%%%%%%
    a_dmcts = Dist_MCTS(index, time, depth, G.node_status, max_depth, G);
    a_coord = Coordinator(n_nodes);
    b_dmcts = Dist_MCTS(n_nodes, time, depth, G.node_status, max_depth, G);
    b_coord = Coordinator(n_nodes);
    for i=1:opt_iters
        a_dmcts = a_dmcts.mcts_search(G, b_coord);
        b_dmcts = b_dmcts.mcts_search(G, a_coord);
        if mod(i,com_mod(big_iter)) == 0 || i ==1
            [a_dmcts, a_coord] = a_dmcts.sample_tree(a_coord);
            [b_dmcts, b_coord] = b_dmcts.sample_tree(b_coord);
        else
            [a_dmcts, ~] = a_dmcts.sample_tree(a_coord);
            [b_dmcts, ~] = b_dmcts.sample_tree(b_coord);
        end
        
        a_dmcts_path = [];
        a_dmcts_path = a_dmcts.exploit(a_dmcts_path);
        b_dmcts_path = [];
        b_dmcts_path = b_dmcts.exploit(b_dmcts_path);
        big_coord_tracker(big_iter,i) = G.evaluate_coord(a_dmcts_path, b_dmcts_path);
    end
end

for  i=1:big_iter
    coord_norm(i,:) = big_coord_tracker(i,:) / max(max(big_coord_tracker(:,end)));
end

coord_mean = mean(coord_norm,1);
coord_std = std(coord_norm, 1);

figure(100)
    plot(coord_mean, 'k');
    hold on;
    errorbar(1:25:2000,coord_mean(1:25:end), coord_std(1:25:end), 'k.');
    axis([0 2000 0 1])

% a_dmcts.down_branch_reward
% b_dmcts.down_branch_reward
% a_dmcts_path = [];
% a_dmcts_path = a_dmcts.exploit(a_dmcts_path);
% b_dmcts_path = [];
% b_dmcts_path = b_dmcts.exploit(b_dmcts_path);

% figure(3)
%     for i=1:n_nodes
%         plot(G.nodes{i}.x, G.nodes{i}.y, 'rx')
%         hold on
%         txt = num2str(i);
%         text(G.nodes{i}.x+0.1, G.nodes{i}.y + 0.1, txt);
%     end
%     grid on
%     for i=1:length(a_dmcts_path)-1
%         plot([G.nodes{a_dmcts_path(i,1)}.x, G.nodes{a_dmcts_path(i+1,1)}.x], [G.nodes{a_dmcts_path(i,1)}.y, G.nodes{a_dmcts_path(i+1,1)}.y], 'g');
%     end
%     plot(G.nodes{a_dmcts_path(1,1)}.x, G.nodes{a_dmcts_path(1,1)}.y, 'go')
%     plot(G.nodes{a_dmcts_path(end,1)}.x, G.nodes{a_dmcts_path(end,1)}.y, 'go')
%     for i=1:length(b_dmcts_path)-1
%         plot([G.nodes{b_dmcts_path(i,1)}.x, G.nodes{b_dmcts_path(i+1,1)}.x], [G.nodes{b_dmcts_path(i,1)}.y, G.nodes{b_dmcts_path(i+1,1)}.y], 'b');
%     end
%     plot(G.nodes{b_dmcts_path(1,1)}.x, G.nodes{b_dmcts_path(1,1)}.y, 'bo')
%     plot(G.nodes{b_dmcts_path(end,1)}.x, G.nodes{b_dmcts_path(end,1)}.y, 'bo')
    
%%%%%%%%%%%%%%%%%% Plots %%%%%%%%%%%%%%%%%%%%%%%%%%
    
%%

for nn = 1:n_nodes
    times = [0];
    probs = [0];
    [times, probs] = a_dmcts.plot_task_probs(nn,times,probs);
    data = [times; probs];
    s_data = sortrows(data',1)';
    cum_prob = zeros(1, length(s_data(2,:)));
    cum_prob(1) = s_data(2,1);
    for i=2:length(s_data(2,:))
        cum_prob(i) = cum_prob(i-1) + s_data(2,i);
    end
    
    figure(4)
        plot(s_data(1,:), cum_prob(:))
        hold on
        grid on
end

