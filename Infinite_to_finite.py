# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:22:29 2025

@author: WONCHAN
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch.nn.functional as F


class Config:
    r = 0.02          # Risk-free rate
    mu = 0.06         # Expected return
    sigma = 0.20      # Volatility
    
    # Income Parameters (Geometric Brownian Motion)
    mu_y = 0.02       # Income growth rate
    sigma_y = 0.10    # Income volatility
    
    # Utility Parameters
    beta = 0.02       # Discount rate
    
    # Simulation & Training
    dt = 0.05         # 
    batch_size = 1024 # 
    n_epochs = 800    # Training epochs 
    lr_actor = 1e-3   # Actor Learning rate 
    lr_critic = 3e-3  # Critic Learning rate
    
    horizons = [30.0] 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4)   # [a0, a1, b0, b1]
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        self.net[-1].weight.data.mul_(0.1)

    def forward(self, t, z, gamma):
        x = t                           # 시간만 입력
        out = self.net(x)
        a0, a1, b0, b1 = torch.chunk(out, 4, dim=1)

        # 투자: Merton + (선형 헤징항)
        merton_val = (cfg.mu - cfg.r) / (gamma * cfg.sigma**2)
        pi_merton = torch.tensor(merton_val, device=cfg.device).float()
        pi_hedging = a0 + a1 * z
        pi_total   = pi_merton + pi_hedging

        # 소비: 선형 + softplus 로 양수화
        c_raw   = b0 + b1 * z
        c_ratio = F.softplus(c_raw) + 1e-6

        return pi_total, c_ratio, pi_hedging
    
class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1) 
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, t, z):
        x = torch.cat([t, z], dim=1)
        val = self.net(x)
        return val

def train_actor_critic(T, actor, critic, opt_actor, opt_critic, rho, gamma):
    actor.train()
    critic.train()
    
    dt = cfg.dt
    n_steps = int(T / dt)
    
    for epoch in range(cfg.n_epochs):
        opt_actor.zero_grad()
        opt_critic.zero_grad()
        
        z0 = torch.FloatTensor(cfg.batch_size, 1).uniform_(0.01, 1.0).to(cfg.device)
        X0 = torch.ones(cfg.batch_size, 1).to(cfg.device)
        y0 = z0 * X0
        
        X, y = X0, y0
        
        dW1 = torch.randn(cfg.batch_size, n_steps).to(cfg.device) * np.sqrt(dt)
        dW_ortho = torch.randn(cfg.batch_size, n_steps).to(cfg.device) * np.sqrt(dt)
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW_ortho 
        
        total_utility = 0
        
        for k in range(n_steps):
            t_curr = torch.ones(cfg.batch_size, 1).to(cfg.device) * (k * dt)
            z = y / (X + 1e-8)
            
            # Actor
            pi, c_ratio, _ = actor(t_curr, z, gamma)
            C = c_ratio * X
            
            # Utility
            discount = torch.exp(-torch.tensor(cfg.beta * k * dt))
            u = (C ** (1 - gamma)) / (1 - gamma)
            total_utility += discount * u * dt
            
            # SDE Update
            drift_X = (cfg.r * X + pi * X * (cfg.mu - cfg.r) - C + y) * dt
            diff_X = pi * X * cfg.sigma * dW1[:, k:k+1]
            X_next = X + drift_X + diff_X
            
            drift_y = cfg.mu_y * y * dt
            diff_y = cfg.sigma_y * y * dW2[:, k:k+1]
            y_next = y + drift_y + diff_y
            
            # Penalty
            mask = (X_next > 1e-4).float()
            X = torch.relu(X_next) * mask + 1e-4
            y = y_next
            
        actor_loss = -torch.mean(total_utility)
        
        t_zero = torch.zeros(cfg.batch_size, 1).to(cfg.device)
        pred_W = critic(t_zero, z0)
        
        scaling_factor = (X0 ** (1-gamma)) / (1-gamma)
        predicted_utility = pred_W * scaling_factor
        target_utility = total_utility.detach() 
        
        critic_loss = nn.MSELoss()(predicted_utility, target_utility)
        
        total_loss = actor_loss + critic_loss
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        
        opt_actor.step()
        opt_critic.step()
        
        if epoch % 100 == 0:
            print(f"Ep {epoch} [Rho {rho} Gam {gamma}] | ActLoss {actor_loss.item():.4f} | CritLoss {critic_loss.item():.4f}")
            
    return actor, critic

def evaluate_networks(actor, critic, gamma):
    actor.eval()
    critic.eval()
    
    z_grid_np = np.linspace(0.01, 1.0, 100)
    z_grid = torch.from_numpy(z_grid_np).float().unsqueeze(1).to(cfg.device)
    t_zero = torch.zeros_like(z_grid).to(cfg.device)
    
    with torch.no_grad():
        pi_total, c_ratio, pi_hedging = actor(t_zero, z_grid, gamma)
        w_val = critic(t_zero, z_grid)
        
    return (z_grid_np,
            pi_total.cpu().numpy(),
            c_ratio.cpu().numpy(),
            w_val.cpu().numpy(),
            pi_hedging.cpu().numpy())

def run_ppgdpo_experiment(T_horizon, gamma_val, rho_val, sigma_y_val,
                          label_str, train_epochs=1000):
    device = cfg.device

    mu    = cfg.mu
    r     = cfg.r
    sigma = cfg.sigma

    dt = T_horizon / 100.0
    if dt > 0.1:
        dt = 0.1

    class SimplePolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(1, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, 1)
            )
            for m in self.fc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0.0)

        def forward(self, z):
            return self.fc(z)

    pi_net = SimplePolicyNet().to(device)
    optimizer = optim.Adam(pi_net.parameters(), lr=1e-3)

    print(f"Start Training: {label_str} (T={T_horizon}, sigma_y={sigma_y_val})")

    for epoch in range(train_epochs + 1):
        optimizer.zero_grad()

        z0 = torch.rand(1000, 1, device=device) * 0.5 + 0.01
        z  = z0.clone().requires_grad_(True)

        pi = pi_net(z)

        ret_term   = pi * (mu - r)
        risk_term  = 0.5 * gamma_val * (sigma ** 2) * (pi ** 2)

        corr = 0.0
        hedge_term = - pi * gamma_val * sigma * sigma_y_val * corr * z

        loss = -torch.mean(ret_term - risk_term - hedge_term)

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"  Ep {epoch:4d} | Loss: {loss.item():.6f}")

    z_grid = torch.linspace(0.01, 0.5, 100, device=device).unsqueeze(1)
    with torch.no_grad():
        pi_pred = pi_net(z_grid).cpu().numpy()

    return z_grid.cpu().numpy(), pi_pred


def experiment_3_convergence_T():
    print("\n=== Experiment 3: Convergence over T ===")
    T_values    = [5.0, 10.0, 20.0, 50.0]
    gamma_fix   = 4.0
    rho_fix     = 0.02
    sigma_y_fix = 0.10

    plt.figure(figsize=(8, 6))

    for T_val in T_values:
        z_res, pi_res = run_ppgdpo_experiment(
            T_horizon=T_val,
            gamma_val=gamma_fix,
            rho_val=rho_fix,
            sigma_y_val=sigma_y_fix,
            label_str=f"T={T_val}",
            train_epochs=1000
        )
        plt.plot(z_res, pi_res, linewidth=2, label=f"T = {T_val:g}")

    plt.title(f"Convergence to Infinite Horizon (gamma={gamma_fix})", fontsize=14)
    plt.xlabel("Income-to-wealth ratio $z$", fontsize=12)
    plt.ylabel("Optimal investment $\\pi(z)$", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def experiment_4_merton_verification():
    print("\n=== Experiment 4: Merton Benchmark (No Income Risk) ===")

    gamma_merton = 4.0
    sigma_stock  = cfg.sigma
    mu_stock     = cfg.mu
    r_rate       = cfg.r

    merton_exact = (mu_stock - r_rate) / (gamma_merton * (sigma_stock ** 2))
    print(f"Theoretical Merton Ratio: {merton_exact:.4f}")

    z_merton, pi_merton = run_ppgdpo_experiment(
        T_horizon=10.0,
        gamma_val=gamma_merton,
        rho_val=0.02,
        sigma_y_val=0.0,
        label_str="Merton Case",
        train_epochs=1000
    )

    plt.figure(figsize=(8, 6))
    plt.plot(z_merton, pi_merton,
             label="P-PGDPO (simulated)", linewidth=2)
    plt.axhline(y=merton_exact,
                linestyle="--", linewidth=2,
                label=f"Merton exact ({merton_exact:.2f})")

    plt.title("Verification: Merton Problem ($\\sigma_y = 0$)", fontsize=14)
    plt.xlabel("Income-to-wealth ratio $z$", fontsize=12)
    plt.ylabel("Optimal investment $\\pi$", fontsize=12)
    plt.ylim(merton_exact - 0.2, merton_exact + 0.2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Merton verification finished.")

def run_all_experiments():
    results_gamma = {}
    results_rho   = {}

    fixed_rho      = 0.02
    gammas_to_test = [0.6, 0.7, 0.8, 0.9]

    print(f"\n=== Exp 1: Varying Gamma (Total {len(gammas_to_test)} runs) ===")
    for gamma_val in gammas_to_test:
        print(f"Running: gamma={gamma_val}...")
        actor  = ActorNet().to(cfg.device)
        critic = CriticNet().to(cfg.device)
        opt_a  = optim.Adam(actor.parameters(),  lr=cfg.lr_actor)
        opt_c  = optim.Adam(critic.parameters(), lr=cfg.lr_critic)

        actor, critic = train_actor_critic(cfg.horizons[0], actor, critic,
                                           opt_a, opt_c, fixed_rho, gamma_val)

        z, pi, c_ratio, W_val, pi_hedging = evaluate_networks(actor, critic, gamma_val)

        results_gamma[gamma_val] = {
            'z': z,
            'pi': pi,
            'c_ratio': c_ratio,
            'W': W_val,
            'pi_hedging': pi_hedging,
        }

    rhos_to_test = [0.02, 0.06, 0.1]
    gammas_for_rho_exp = [0.6, 0.9]

    print(f"\n=== Exp 2: Varying Rho (Total {len(gammas_for_rho_exp)*len(rhos_to_test)} runs) ===")
    for gamma_val in gammas_for_rho_exp:
        results_rho[gamma_val] = {}
        for rho_val in rhos_to_test:
            print(f"Running: gamma={gamma_val}, rho={rho_val}...")
            actor = ActorNet().to(cfg.device)
            critic = CriticNet().to(cfg.device)
            opt_a = optim.Adam(actor.parameters(), lr=cfg.lr_actor)
            opt_c = optim.Adam(critic.parameters(), lr=cfg.lr_critic)
        
            actor, critic = train_actor_critic(
                cfg.horizons[0], actor, critic, opt_a, opt_c, rho_val, gamma_val
            )
            z, pi, c_ratio, W_val, pi_hedging = evaluate_networks(actor, critic, gamma_val)
            results_rho[gamma_val][rho_val] = {
                'z': z,
                'pi': pi,
                'c_ratio': c_ratio,
                'W': W_val,
                'pi_hedging': pi_hedging,
            }

    print("\nGeneratng Plots...")
    sns.set_style("whitegrid")

    # Figure 1: Value function W(z)
    plt.figure(figsize=(8, 6))
    for gamma_val in gammas_to_test:
        res = results_gamma[gamma_val]
        plt.plot(res['z'], res['W'], label=f'$\\gamma$: {gamma_val}', 
                 linewidth=(3 if gamma_val==0.6 else 1.5), 
                 linestyle=('--' if gamma_val==0.6 else '-.' if gamma_val==0.7 else ':' if gamma_val==0.8 else '-'))
    plt.title("Value function $W(z)$", fontsize=14)
    plt.xlabel("Income-to-wealth ratio $z$", fontsize=12)
    plt.ylabel("Value function $W(z)$", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Figure 2: Optimal consumption C(z)
    plt.figure(figsize=(8, 6))
    for gamma_val in gammas_to_test:
        res = results_gamma[gamma_val]
        plt.plot(res['z'], res['c_ratio'], label=f'$\\gamma$: {gamma_val}', 
                 linewidth=(3 if gamma_val==0.6 else 1.5), 
                 linestyle=('--' if gamma_val==0.6 else '-.' if gamma_val==0.7 else ':' if gamma_val==0.8 else '-'))
    plt.title("Optimal consumption $C(z)$", fontsize=14)
    plt.xlabel("Income-to-wealth ratio $z$", fontsize=12)
    plt.ylabel("Optimal consumption $C(z)$", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Figure 3: Optimal investment w(z)
    plt.figure(figsize=(8, 6))
    for gamma_val in gammas_to_test:
        res = results_gamma[gamma_val]
        plt.plot(res['z'], res['pi'], label=f'$\\gamma$: {gamma_val}', 
                 linewidth=(3 if gamma_val==0.6 else 1.5), 
                 linestyle=('--' if gamma_val==0.6 else '-.' if gamma_val==0.7 else ':' if gamma_val==0.8 else '-'))
    plt.title("Optimal investment $\\bar{\\omega}(z)$", fontsize=14)
    plt.xlabel("Income-to-wealth ratio $z$", fontsize=12)
    plt.ylabel("Optimal investment $\\bar{\\omega}(z)$", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Figure 4 & 5: Consumption (rho varied)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    styles = {0.02: '--', 0.06: ':', 0.1: '-'}
    colors = {0.02: 'blue', 0.06: 'green', 0.1: 'red'}

    for i, gamma_val in enumerate(gammas_for_rho_exp):
        ax = axes[i]
        for rho_val in rhos_to_test:
            res = results_rho[gamma_val][rho_val]
            ax.plot(res['z'], res['c_ratio'], label=f'$\\rho$: {rho_val}', 
                    linestyle=styles[rho_val], color=colors[rho_val], linewidth=2)
        ax.set_title(f"Optimal consumption $C(z)$ for $\\gamma={gamma_val}$", fontsize=14)
        ax.set_xlabel("Income-to-wealth ratio $z$", fontsize=12)
        ax.set_ylabel("Optimal consumption $C(z)$", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Figure 6 & 7: Investment (rho varied)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, gamma_val in enumerate(gammas_for_rho_exp):
        ax = axes[i]
        for rho_val in rhos_to_test:
            res = results_rho[gamma_val][rho_val]
            ax.plot(res['z'], res['pi'], label=f'$\\rho$: {rho_val}', 
                    linestyle=styles[rho_val], color=colors[rho_val], linewidth=2)
        ax.set_title(f"Optimal investment $\\bar{{\\omega}}(z)$ for $\\gamma={gamma_val}$", fontsize=14)
        ax.set_xlabel("Income-to-wealth ratio $z$", fontsize=12)
        ax.set_ylabel("Optimal investment $\\bar{{\\omega}}(z)$", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Done!")

if __name__ == "__main__":
    start_time = time.time()

    run_all_experiments()

    experiment_3_convergence_T()
    experiment_4_merton_verification()

    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time)/60:.2f} minutes")

