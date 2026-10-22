// Command probe é o cliente do gateway (internal/udscache) pedindo STATS ao
// cache pelo socket, com as variáveis do gateway: FLOW_CACHE_SOCKET_PATH e
// FLOW_CACHE_SOCKET_OWNER_UID (o dono esperado do socket e do processo do
// outro lado). Entra no checkout do flow-gateway-go em cmd/gateway-probe/
// (internal/ só se importa de dentro do módulo); o job da imagem, em
// .github/workflows/docker.yml, roda o binário no container do serviço `ping`
// de deploy/compose/compose.yaml. Sai 0 com a resposta, 2 com o socket
// recusado pelo cliente (ErrUntrustedSocket) e 1 com qualquer outra falha.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/Educacao-Solidaria/flow-gateway-go/internal/udscache"
)

func main() { os.Exit(run()) }

func run() int {
	path := os.Getenv("FLOW_CACHE_SOCKET_PATH")
	uid, err := strconv.Atoi(os.Getenv("FLOW_CACHE_SOCKET_OWNER_UID"))
	if path == "" || err != nil {
		fmt.Fprintln(os.Stderr, "defina FLOW_CACHE_SOCKET_PATH e FLOW_CACHE_SOCKET_OWNER_UID")
		return 1
	}
	client := udscache.New(path, uid, nil)
	defer client.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stats, err := client.Stats(ctx)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		if errors.Is(err, udscache.ErrUntrustedSocket) {
			return 2
		}
		return 1
	}
	out, err := json.Marshal(stats)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	fmt.Println(string(out))
	return 0
}
