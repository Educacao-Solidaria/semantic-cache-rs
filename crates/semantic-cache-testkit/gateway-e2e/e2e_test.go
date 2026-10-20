// Ponta a ponta: o cliente do gateway (internal/udscache) contra os binários
// Rust do semantic-cache-rs, por socket Unix de verdade. Mora no
// semantic-cache-rs (crates/semantic-cache-testkit/gateway-e2e/); o job
// "e2e (rust ⇄ go)" de lá o copia para internal/e2erust/ de um checkout fixado
// do flow-gateway-go, porque internal/ só se importa de dentro do módulo.
// SEMCACHE_E2E_SERVER: o binário de produção. SEMCACHE_E2E_REFERENCE: o
// reference-server do testkit (LOOKUP e STORE, até o PR 52/100 chegar a ele).
package e2erust

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/Educacao-Solidaria/flow-gateway-go/internal/ipc"
	"github.com/Educacao-Solidaria/flow-gateway-go/internal/udscache"
	"github.com/Educacao-Solidaria/flow-gateway-go/pkg/client/cache"
)

// patience é o prazo de toda espera: runner lento vira falha, não teste pendurado.
const patience = 10 * time.Second

// dim é a dimensão do índice do servidor de referência.
const dim = 64

// rust é um binário do semantic-cache-rs no ar.
type rust struct {
	cmd  *exec.Cmd
	done chan struct{} // fechado quando o processo sai; err diz como
	err  error
}

// start sobe o binário da variável env com o socket em path e espera o PONG.
func start(t *testing.T, env, path string, args ...string) *rust {
	t.Helper()
	bin := os.Getenv(env)
	require.NotEmpty(t, bin, "%s: caminho do binário (o job e2e do semantic-cache-rs define)", env)
	cmd := exec.Command(bin, args...)
	cmd.Env = append(os.Environ(), "SEMCACHE_IPC_SOCKET_PATH="+path)
	cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
	require.NoError(t, cmd.Start())
	r := &rust{cmd: cmd, done: make(chan struct{})}
	go func() { r.err = cmd.Wait(); close(r.done) }()
	t.Cleanup(func() { _ = cmd.Process.Kill(); <-r.done })
	up := func() bool { return ping(path) == nil }
	require.Eventually(t, up, patience, 10*time.Millisecond, "%s não respondeu ao PING", env)
	return r
}

// stop manda SIGTERM e espera a saída limpa, com código 0.
func (r *rust) stop(t *testing.T) {
	t.Helper()
	require.NoError(t, r.cmd.Process.Signal(syscall.SIGTERM))
	select {
	case <-r.done:
		require.NoError(t, r.err, "saída do servidor")
	case <-time.After(patience):
		t.Fatal("o servidor não saiu com SIGTERM")
	}
}

// ping manda um PING cru (o cliente do gateway não tem PING) e espera o PONG.
func ping(path string) error {
	nc, err := net.DialTimeout("unix", path, patience)
	if err != nil {
		return err
	}
	defer nc.Close()
	_ = nc.SetDeadline(time.Now().Add(patience))
	raw, _ := ipc.Append(nil, ipc.Frame{Kind: ipc.KindPing, RequestID: 7})
	if _, err = nc.Write(raw); err != nil {
		return err
	}
	f, err := ipc.ReadFrame(nc)
	if err == nil && (f.Kind != ipc.KindPong || f.RequestID != 7) {
		err = fmt.Errorf("resposta %#02x (id %d) ao PING", f.Kind, f.RequestID)
	}
	return err
}

// socketPath é um caminho curto num diretório 0700 (o t.TempDir() leva o nome
// do teste, e o caminho de um socket cabe em 108 bytes).
func socketPath(t *testing.T) string {
	dir, err := os.MkdirTemp("", "e2e")
	require.NoError(t, err)
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	return filepath.Join(dir, "c.sock")
}

// mix é a·e_i + b·e_(i+1) em R^dim: com a² + b² = 1, o vetor unitário com
// cosseno a com e_i e b com e_(i+1).
func mix(i int, a, b float64) []float32 {
	v := make([]float32, dim)
	v[i%dim], v[(i+1)%dim] = float32(a), float32(b)
	return v
}

// parallel roda f(0), …, f(n-1) em goroutines e falha no primeiro erro.
func parallel(t *testing.T, n int, f func(int) error) {
	t.Helper()
	errs := make(chan error, n)
	for i := range n {
		go func() { errs <- f(i) }()
	}
	for range n {
		require.NoError(t, <-errs)
	}
}

// O fluxo do gateway contra o servidor de referência: STORE, LOOKUP hit e miss
// de muitas goroutines em pipeline numa conexão só, STATS e as recusas.
func TestReferenceServer(t *testing.T) {
	path := socketPath(t)
	start(t, "SEMCACHE_E2E_REFERENCE", path, path, strconv.Itoa(dim))
	c := udscache.New(path, -1, nil)
	t.Cleanup(func() { _ = c.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), patience)
	defer cancel()

	stats, err := c.Stats(ctx)
	require.NoError(t, err)
	require.Equal(t, cache.StatsResult{}, *stats)

	const entries, rounds = 32, 20
	answer := func(i int) string { return fmt.Sprintf("resposta %d: olá ≠ \"aspas\"\n", i) }
	parallel(t, entries, func(i int) error {
		return c.Store(ctx, fmt.Sprintf("prompt %d", i), answer(i), mix(i, 1, 0), 0)
	})
	// Por rodada, um hit (cosseno 0,98 com e_i) e um miss (0,71 com e_i e
	// e_(i+1)) por entrada, todos ao mesmo tempo.
	near, between := math.Sqrt(1-0.98*0.98), math.Sqrt2/2
	parallel(t, entries*rounds, func(n int) error {
		i := n % entries
		hit, err := c.Lookup(ctx, "p", mix(i, 0.98, near), 0.9)
		if err != nil {
			return err
		}
		if !hit.Hit || hit.CachedResponse != answer(i) || math.Abs(float64(hit.Score)-0.98) > 1e-4 {
			return fmt.Errorf("entrada %d: hit esperado, veio %+v", i, *hit)
		}
		miss, err := c.Lookup(ctx, "p", mix(i, between, between), 0.9)
		if err != nil {
			return err
		}
		if miss.Hit || miss.CachedResponse != "" || math.Abs(float64(miss.Score)-between) > 1e-4 {
			return fmt.Errorf("entrada %d: miss esperado, veio %+v", i, *miss)
		}
		return nil
	})
	stats, err = c.Stats(ctx)
	require.NoError(t, err)
	lookups := uint64(entries * rounds)
	want := cache.StatsResult{TotalEntries: entries, HitCount: lookups, MissCount: lookups, HitRatio: 0.5}
	require.Equal(t, want, *stats)

	// Pedido recusado volta ERROR bad_request sem eco do pedido, e a conexão segue.
	var se *udscache.ServerError
	_, err = c.Lookup(ctx, "p", make([]float32, dim+1), 0.9)
	require.ErrorAs(t, err, &se)
	require.Equal(t, [2]string{ipc.CodeBadRequest, "dimensão 65 ≠ 64"}, [2]string{se.Code, se.Message})
	err = c.Store(ctx, "p", "r", make([]float32, dim), 0) // norma zero
	require.ErrorAs(t, err, &se)
	require.Equal(t, ipc.CodeBadRequest, se.Code)
	stats, err = c.Stats(ctx)
	require.NoError(t, err)
	require.Equal(t, want, *stats, "recusa não conta nem guarda")
}

// O binário de produção: socket 0600 do próprio uid, STATS pelo cliente do
// gateway, socket de outro dono recusado, frame corrompido, SIGTERM e a volta
// do mesmo cliente quando o servidor sobe de novo no mesmo caminho.
func TestServerBinary(t *testing.T) {
	path := socketPath(t)
	server := start(t, "SEMCACHE_E2E_SERVER", path)
	fi, err := os.Lstat(path)
	require.NoError(t, err)
	require.Equal(t, os.ModeSocket, fi.Mode().Type())
	require.Equal(t, os.FileMode(0o600), fi.Mode().Perm())

	c := udscache.New(path, -1, nil)
	t.Cleanup(func() { _ = c.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 4*patience)
	defer cancel()
	stats, err := c.Stats(ctx)
	require.NoError(t, err)
	require.Zero(t, stats.TotalEntries)

	// Dono esperado diferente: o cliente nem disca.
	other := udscache.New(path, os.Getuid()+1, slog.New(slog.NewTextHandler(io.Discard, nil)))
	_, err = other.Stats(ctx)
	require.ErrorIs(t, err, udscache.ErrUntrustedSocket)
	require.NoError(t, other.Close())

	// CRC trocado: ERROR bad_frame com o mesmo request_id, e o servidor fecha.
	nc, err := net.DialTimeout("unix", path, patience)
	require.NoError(t, err)
	defer nc.Close()
	require.NoError(t, nc.SetDeadline(time.Now().Add(patience)))
	raw, _ := ipc.Append(nil, ipc.Frame{Kind: ipc.KindPing, RequestID: 9})
	raw[len(raw)-1] ^= 1
	_, err = nc.Write(raw)
	require.NoError(t, err)
	f, err := ipc.ReadFrame(nc)
	require.NoError(t, err)
	var reply udscache.ServerError
	require.NoError(t, json.Unmarshal(f.Payload, &reply))
	require.Equal(t, [3]any{ipc.KindError, uint32(9), ipc.CodeBadFrame}, [3]any{f.Kind, f.RequestID, reply.Code})
	_, err = ipc.ReadFrame(nc)
	require.ErrorIs(t, err, io.EOF)

	// SIGTERM: sai com 0 e leva o socket; o cliente passa a ver o cache indisponível.
	server.stop(t)
	_, err = os.Lstat(path)
	require.ErrorIs(t, err, os.ErrNotExist)
	unavailable := func() bool { _, err := c.Stats(ctx); return errors.Is(err, udscache.ErrUnavailable) }
	require.Eventually(t, unavailable, patience, 10*time.Millisecond)

	// De volta no mesmo caminho, o mesmo cliente redisca sozinho depois do backoff.
	start(t, "SEMCACHE_E2E_SERVER", path)
	back := func() bool { _, err := c.Stats(ctx); return err == nil }
	require.Eventually(t, back, patience, 20*time.Millisecond)
}
