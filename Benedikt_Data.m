clear all %#ok<CLALL>
close all
addpath('.\quadriga_src\')
% This file generates 4D channels in spatial, frequency and time domain.
% If set_clustered is true, the users per scenario are all inside a circle
% of 5m radius in order to sample the covariance matrix of the channels.

%sim_server = true;
rep_factor = 500; %number of different scenarios
no_of_UEs_perScenario = 220;
%no_cov_calc = 1000; %number of channels that are used for calculating the
%channel cov matrix and not being used for channel generation
n_symbols = 1; %number of times symbols per slot
veloc = 3; %velocity of users in km/h
rand_speed  = false; %choose random speed in between 0 - veloc km/h
duration_per_slot = 1e-3;
n_slots = 1;
symbol_duration = duration_per_slot/n_symbols;
max_dist_to_BS = 500; % multiple of 100
overall_norm2 = 0;

%dir = 'results/Quadriga_4D';

%if sim_server
%%    addpath(genpath('~/lrz-nashome/QuaDriGa_2020.11.03_v2.4.0'));
%else
  %  addpath(genpath('~/Documents/02_Promotion/Quadriga/QuaDriGa_2020.11.03_v2.4.0'));
%end

bandwidth = 120e3;    
carriers = 1;
subcarrier_spacing = bandwidth / carriers;
K = 1;  % polarization indicator: K = 1 -> vertical polarization only (page 33)
bs_no_vertical_elements = 1;
%bs_no_horizontal_elements = 64;
bs_no_horizontal_elements = 32;
ms_no_vertical_elements = 1;
%ms_no_horizontal_elements = 2;
ms_no_horizontal_elements = 1;

bs_mult = bs_no_horizontal_elements * bs_no_vertical_elements;
ms_mult = ms_no_horizontal_elements * ms_no_vertical_elements;
n_mult = bs_mult * ms_mult;

s = qd_simulation_parameters ;
s.show_progress_bars = 0;
s.center_frequency = 6e9;
s.sample_density = 1.2;
s.use_absolute_delays = 1;  % include delay of the LOS path
H_4d = zeros(rep_factor,no_of_UEs_perScenario,bs_mult,ms_mult,carriers,n_slots*n_symbols);
% for debugging replace parfor by for
%parpool(4)
%Cov = zeros(n_slots*n_symbols*carriers, n_slots*n_symbols*carriers);
%Cov_time = zeros(n_slots*n_symbols, n_slots*n_symbols);
%Cov_freq = zeros(carriers, carriers);
%parpool(2)
for i_process = 1:rep_factor  
    disp(i_process)
    l = qd_layout(s);
    l.name = 'custom_layout_for_testing';
    % base station
    l.tx_position(3) = 25;  % 25m base station height
    l.tx_array = qd_arrayant('3gpp-3D', bs_no_vertical_elements, bs_no_horizontal_elements, s.center_frequency, K);
    % mobile terminals
    
    %l.rx_array = qd_arrayant('3gpp-3D', ms_no_vertical_elements, ms_no_horizontal_elements, s.center_frequency, K);
    
    % create a uniform linear array with dipole antennas
    l.rx_array = qd_arrayant('omni');
    l.rx_array.no_elements = ms_no_horizontal_elements;
    for i = 1:l.rx_array.no_elements
        l.rx_array.element_position(:, i) = [(i-1)*s.wavelength/2; 0; 0];
    end
    
    l.no_rx = no_of_UEs_perScenario;
    l.randomize_rx_positions(max_dist_to_BS , 1.5 , 1.5 , 0);  % <- this 0 makes the MS stationary
    phi = 2*pi/3*rand(l.no_rx,1) + 1/6 *pi;
    r = rand(no_of_UEs_perScenario,1)*465 + 35;
    x = sin(phi);
    y = cos(phi);
    M = [x,y];
    Mat = r.* M;
    Mat(:,3) = 1.5; % 1.5m height
    l.rx_position = Mat';
    %l.tx_array.rotate_pattern(180, 'z');    
    
    %if set_clustered == true
    %    Mat = zeros(3,l.no_rx);
    %    init_loc = l.rx_position(:,1);
    %    Mat(1:2,:) = rand_circ(l.no_rx, init_loc(1),init_loc(2), 5);
    %    Mat = Mat';
    %end
    
    %figure;
    %scatter(Mat(:,1),Mat(:,2),Mat(:,3));

    % Set random height of the users
    floor = randi(5,1,l.no_rx) + 3;                   % Number of floors in the building
    for n = 1 : l.no_rx
        floor( n ) =  randi( floor( n ) );                 % Floor level of the UE
    end
    zcoord = 3*(floor-1) + 1.5;           % Height in meters    
    
    
    Mat(:,3) = zcoord';
    l.rx_position = Mat';
    
    %l.visualize;
    %for i_nosnap = 1:l.no_rx
    %    l.rx_track(1,i_nosnap).no_snapshots = 1;
    %end
    
%     for i_rotate = 1:l.no_r
%        random_angle = rand(1)*360;
%        l.rx_array(1,i_rotate).rotate_pattern(random_angle,'z');
%     end

    for i_track = 1:l.no_rx
        random_angle = rand(1)*2*pi;
        if rand_speed
            veloc_rand = veloc / 3.6 * rand(1);
            length_per_slot = duration_per_slot * veloc_rand;
            t = qd_track('linear',n_slots*length_per_slot,random_angle);
            t.name = sprintf('MS-%i-%i',i_process,i_track);
            t.initial_position = l.rx_position(:,i_track);
            t.set_speed(veloc_rand) %terminal speed in m/s
        else
            length_per_slot = duration_per_slot * veloc / 3.6;
            t = qd_track('linear',n_slots*length_per_slot,random_angle);
            t.name = sprintf('MS-%i-%i',i_process,i_track);
            t.initial_position = l.rx_position(:,i_track);
            t.set_speed(veloc / 3.6) %terminal speed in m/s
        end

        [~, l.rx_track(:,i_track)] = interpolate(t.copy,'time',symbol_duration);
    end
    %l.visualize;
    
    
    % plot if number of users is small
    %figure;
    %if l.no_rx < 51
    %    l.visualize;
    %end
        %disp('Set Scenario')
    indoor_rx = l.set_scenario('3GPP_38.901_UMa', [], [], 0.8);
    l.rx_position(3,~indoor_rx) = 1.5;             	% Set outdoor-users to 1.5 m height
    % generate channel coefficients
    [h_channel, h_builder] = l.get_channels();
    % the coefficients can be found as h_channel.coeff: [no_rxant, no_txant, no_path, no_snap]
    %test = h_channel.coeff;
    % transform into uplink channel
    h_channel.swap_tx_rx();
    
    H = cell(l.no_rx, 1);
    no_los = 0;
    no_nlos = 0;
    powers_all = zeros(l.no_rx, 1);
    H_per = zeros(l.no_rx,bs_mult,ms_mult,carriers,n_slots*n_symbols);
    %Cov_per = zeros(n_slots*n_symbols,carriers,carriers);
    for ms = 1:l.no_rx
        % transform channel to frequency domain
        chan = h_channel(ms).fr(bandwidth, carriers,1:n_slots*n_symbols);
        H{ms}.channel = chan;
        
        % MS position
        H{ms}.pos = h_channel(ms).rx_position;
        % find MS scenario (LOS vs. NLOS)
        if strfind(l.rx_track(ms).scenario{1}, 'NLOS')
            H{ms}.los = 0;
            no_nlos = no_nlos + 1;
        else
            H{ms}.los = 1;
            no_los = no_los + 1;
        end
        H{ms}.pg = h_channel(ms).par.pg_parset;
        
        %Normalizing path loss (see docu p.204)
        norm_factor = sqrt(10^(0.1*H{ms}.pg));
        H{ms}.channel = H{ms}.channel ./ norm_factor;
        %H{ms}.channel = reshape(H{ms}.channel,[1,n_mult,carriers]);
        %n_time = n_slots*n_symbols;
        chan_vect = H{ms}.channel(:);
        overall_norm2 = overall_norm2 + norm(chan_vect)^2;
        
        H_per(ms,:,:,:,:) = H{ms}.channel;
    end

    %l.visualize;
    H_4d(i_process,:,:,:,:,:) = H_per;
    %disp(['Number of LOS channels: ', num2str(no_los)]):q!
    %disp(['Number of NLOS channels: ', num2str(no_nlos)])
    
end
clear chan;
%Normalization factor for the whole dataset:
overall_norm2 = sqrt(overall_norm2 / (no_of_UEs_perScenario) / rep_factor);
H_all = zeros(rep_factor*no_of_UEs_perScenario,bs_mult,ms_mult,carriers,n_slots*n_symbols);
count = 1;
for iter=1:rep_factor
    for n_ue=1:no_of_UEs_perScenario
        H_all(count,:,:,:,:) = sqrt(bs_mult*ms_mult*carriers*n_slots*n_symbols)*H_4d(iter,n_ue,:,:,:,:) ./ overall_norm2;
        count = count + 1;
    end
end
clear H_4d; 
t = datestr(now,'mm-dd-yyyy_HH-MM-SS');

filename_all = fullfile(dir,sprintf('%s_%dUEs_500m_%dx%dBS_%dx%dMS_%dcarr_%dsymb_%dkmh_diff=%d.mat',t,rep_factor*no_of_UEs_perScenario,bs_no_vertical_elements,bs_no_horizontal_elements,ms_no_vertical_elements,ms_no_horizontal_elements,carriers,n_symbols,veloc,rand_speed));
   
% permute randomly
%idx = randperm(rep_factor * no_of_UEs_perScenario);
%H_all = H_all(idx, :, :, :, :);
%Cov_all = single(Cov_all(idx, :, :, :));

%pre-processing
%H_all = squeeze(H_all);
%Cov = squeeze(Cov);

test = sum(abs(H_all(:)).^2) / numel(H_all);

save(filename_all, 'H_all', '-v7.3');
%save(filename_cov, 'Cov', '-v7.3');
%save(filename_cov_time, 'Cov_time', '-v7.3');
%save(filename_cov_freq, 'Cov_freq', '-v7.3');
