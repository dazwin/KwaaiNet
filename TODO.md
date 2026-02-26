# TODO

## Networking

- [ ] **Fix relay fallback** — `metro@kwaai` (peer `...5bZ251`) connects via p2p-circuit relay through `76.91.214.120` instead of direct on configured public IP `75.141.127.202:8080`. Node should establish a direct connection. Investigate NAT traversal / port forwarding and `announceAddrs` config.
