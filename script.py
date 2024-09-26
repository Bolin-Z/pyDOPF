from distopf.networks import load_net_9, load_net_30, load_net_118_a, load_net_118_b
import pickle

# with open('./power_case/case9', 'wb') as f:
#     net = load_net_9()
#     pickle.dump(net, f)

# with open('./power_case/case30', 'wb') as f:
#     net = load_net_30()
#     pickle.dump(net, f)

with open('./power_case/case118a', 'wb') as f:
    net = load_net_118_a()
    pickle.dump(net, f)

with open('./power_case/case118b', 'wb') as f:
    net = load_net_118_b()
    pickle.dump(net, f)