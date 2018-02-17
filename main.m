close all
clc

com_mod = [1,10,50,100,250,500,750,1000];
dropout_rate = [0.0, 0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]; % p com drops
min_sampling = [0.001,0.005,0.01,0.05,0.1,0.25,0.5, 0.75, 0.9, 0.95, 0.99, 1.0];
alpha_set = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99,1.0];
beta_set = [0.141, 0.5, 1.0, 1.41, 2.0, 2.5, 5.0, 10.0];
gamma_set = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]; 
rollout_set = [0,1,2,3,5,7,10];
init_iters = 500;
opt_iters = 2500;
num_maps = 20;
n_nodes = 20;

test_var = min_sampling;
test_var_name = 'min sampling';
n_test_var = length(test_var);

big_coord_tracker = zeros(num_maps,n_test_var,init_iters + opt_iters);
a_down_branch = zeros(num_maps, n_test_var, init_iters + opt_iters);
b_down_branch = zeros(num_maps, n_test_var, init_iters + opt_iters);

for map_iter = 1:num_maps
    clearvars -except big_coord_tracker test_iter test_var opt_iters map_iter rollout_test num_maps init_iters test_var_name n_nodes n_test_var
    total_iters = init_iters + opt_iters;
    parfor test_iter = 1:n_test_var
        coord_temp = zeros(1,total_iters);
        a_temp = zeros(1,total_iters);
        b_temp = zeros(1,total_iters);
        rng(map_iter);
        G = World(n_nodes);
        percent_done = (map_iter-1)/num_maps + (test_iter-1)/(num_maps*length(test_var))
        if true
            if strcmp(test_var_name, 'alpha')
                G.dmcts_alpha = test_var(test_iter); % gradient descent on agent policy
            else
                G.dmcts_alpha = 0.95;
            end
            if strcmp(test_var_name,'beta')
                G.dmcts_beta = test_var(test_iter); % explore vs exploit
            else
                G.dmcts_beta = 1.41; % explore vs exploit
            end
            if strcmp(test_var_name,'gamma')
                G.dmcts_gamma = test_var(test_iter); % explore vs exploit
            else
                G.dmcts_gamma = 0.999; % explore vs exploit
            end
            
            if strcmp(test_var_name,'dropout rate')
                dropout_rate = test_var(test_iter); % p com drops
            else
                dropout_rate = 0.05; % p com drops
            end

            if strcmp(test_var_name,'com mod')
                com_mod = test_var(test_iter); % how many do I transmit
            else
                com_mod = 10; % how many do I transmit
            end
            
            if strcmp(test_var_name,'min sampling')
                G.dmcts_min_sampling_threshold = test_var(test_iter); % how far down the tree do I sample
            else
                G.dmcts_min_sampling_threshold = 0.25; % how far down the tree do I sample
            end

            if strcmp(test_var_name,'rollout')
                G.dmcts_max_rollout_depth = test_var(test_iter); %rollout_test(test_iter);
            else
                G.dmcts_max_rollout_depth = 1; %rollout_test(test_iter);
            end
       end
        
        time = 0;
        depth = 1;
        max_depth = n_nodes/2 - 1;

        %%%%%%%%%%%%%%%%%% Dist-MCTS Test %%%%%%%%%%%%%%%%%%%%%%
        a_dmcts = Dist_MCTS(1, -1, time, depth, G.node_status, max_depth, G);
        a_coord = Coordinator(n_nodes);
        b_dmcts = Dist_MCTS(n_nodes, -1, time, depth, G.node_status, max_depth, G);
        b_coord = Coordinator(n_nodes);
        c_coord = Coordinator(n_nodes);
        for i=1:total_iters
            a_dmcts = a_dmcts.mcts_search(G, b_coord);
            b_dmcts = b_dmcts.mcts_search(G, a_coord);

            if i >= init_iters     
                if rand() > dropout_rate && mod(i,com_mod) == 0
                    [a_dmcts, a_coord] = a_dmcts.sample_tree(a_coord);
                    [b_dmcts, b_coord] = b_dmcts.sample_tree(b_coord);
                else
                    [a_dmcts, ~] = a_dmcts.sample_tree(a_coord);
                    [b_dmcts, ~] = b_dmcts.sample_tree(b_coord);
                end
            end

            a_dmcts_path = [];
            a_dmcts_path = a_dmcts.exploit(a_dmcts_path);
            b_dmcts_path = [];
            b_dmcts_path = b_dmcts.exploit(b_dmcts_path);
            coord_temp(i) = G.evaluate_coord(a_dmcts_path, b_dmcts_path);
            a_temp(i) = a_dmcts.down_branch_reward;
            b_temp(i) = b_dmcts.down_branch_reward;
        end
        big_coord_tracker(map_iter,test_iter,:) = coord_temp;
        a_down_branch(test_iter, :) = a_temp;
        b_down_branch(test_iter, :) = b_temp;
    end
end

coord_norm = zeros(num_maps, length(test_var), init_iters + opt_iters);
coord_mean = zeros(length(test_var),init_iters + opt_iters);
coord_std = zeros(length(test_var), init_iters + opt_iters);
max_reward = zeros(1,num_maps);
for  i=1:length(test_var)
    for j=1:num_maps
        max_reward(j) = max(max(big_coord_tracker(j,:,1:end)));
        coord_norm(j,i,:) = big_coord_tracker(j,i,:) / 1;%max_reward(j);
    end
    coord_mean(i,:) = mean(coord_norm(:,i,:),1);
    if num_maps > 1
        coord_std(i,:) = std(squeeze(coord_norm(:,i,:)),1)/sqrt(num_maps);
    end
end
for  i=1:length(test_var)
    figure(99)
        title(test_var_name);
        if num_maps == 1
            plot(coord_mean(i,:), 'DisplayName', num2str(test_var(i)));
        else
            errorbar(1:25:total_iters,coord_mean(i,1:25:end), coord_std(i,1:25:end), 'DisplayName', num2str(test_var(i)));
        end
        hold on
        %for j=1:num_maps
        %    plot(squeeze(coord_norm(j,i,:)), 'r');
        %end
        hold on
        grid on
end
legend('show')

for  i=1:length(test_var)
     figure(100)
        title('A Down Branch');
        plot(1:opt_iters + init_iters,a_down_branch(i,:),'DisplayName', num2str(test_var(i)));
        hold on;
        grid on;
 end
legend('show')

for  i=1:length(test_var)
      figure(101)
        title('B Down Branch');
        plot(1:opt_iters + init_iters,b_down_branch(i,:),'DisplayName', num2str(test_var(i)));
        hold on;
        grid on;
end
legend('show')
 
 for  i=1:length(test_var)
      figure(102)
        title('A+B Down Branch');
        plot(1:opt_iters + init_iters,a_down_branch(i,:) + b_down_branch(i,:),'DisplayName', num2str(test_var(i)));
        hold on;
        grid on;
end
legend('show')


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

% for nn = 1:n_nodes
%     times = [0];
%     probs = [0];
%     [times, probs] = a_dmcts.plot_task_probs(nn,times,probs);
%     data = [times; probs];
%     s_data = sortrows(data',1)';
%     cum_prob = zeros(1, length(s_data(2,:)));
%     cum_prob(1) = s_data(2,1);
%     for i=2:length(s_data(2,:))
%         cum_prob(i) = cum_prob(i-1) + s_data(2,i);
%     end
%     
%     figure(4)
%         plot(s_data(1,:), cum_prob(:))
%         hold on
%         grid on
% end


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
