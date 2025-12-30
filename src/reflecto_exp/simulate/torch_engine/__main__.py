import matplotlib.pyplot as plt
import torch

from reflecto_exp.simulate.torch_engine import simulate_reflectivity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q = torch.linspace(0.01, 0.3, 100).view(1, -1).to(device)

thickness = torch.tensor([[300.0]]).to(device)
roughness = torch.tensor([[3.0, 2.0]]).to(device)
sld = torch.tensor([[20.0, 10.0, 0.0]]).to(device)

# 2. 시뮬레이션 실행
R = simulate_reflectivity(
    q=q,
    thickness=thickness,
    roughness=roughness,
    sld=sld,
    dq_q=torch.tensor([[0.02]]).to(device)
)

R_numpy = R.detach().cpu().numpy()
q_numpy = q.detach().cpu().numpy()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(q_numpy[0], R_numpy[0])
ax.set_yscale("log")
plt.show()
