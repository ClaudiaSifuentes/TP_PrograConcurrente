package nn

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	mat "neural_network/modules/mymat"
)

type Config struct {
	Eta       float64
	BatchSize int
	Epochs    int
}

type MLP struct {
	weights   []*mat.Dense
	biases    []*mat.Dense
	numLayers int
	config    Config
}

// applySigmoid applies σ(z)
func applySigmoid(_, _ int, v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-v))
}

// applySigmoidPrime expects v = σ(z), so computes σ(z)*(1-σ(z))
func applySigmoidPrime(_, _ int, v float64) float64 {
	return v * (1.0 - v)
}

// addBias adds the bias row-vector to each row of the pre-activation matrix
func addBias(b *mat.Dense) func(_, col int, v float64) float64 {
	return func(_, col int, v float64) float64 {
		return v + b.At(0, col)
	}
}

func oneHotDecode(row []float64) int {
	maxIdx := 0
	maxVal := row[0]
	for i, v := range row {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

func prediction(row []float64) int {
	return oneHotDecode(row)
}

func NewMLP(sizes []int, config Config) *MLP {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	weights := make([]*mat.Dense, len(sizes)-1)
	biases := make([]*mat.Dense, len(sizes)-1)

	for i := 1; i < len(sizes); i++ {
		w := mat.NewDense(sizes[i-1], sizes[i], nil)
		b := mat.NewDense(1, sizes[i], nil)
		for j := 0; j < sizes[i-1]; j++ {
			for k := 0; k < sizes[i]; k++ {
				w.Set(j, k, r.NormFloat64())
			}
		}
		for k := 0; k < sizes[i]; k++ {
			b.Set(0, k, r.NormFloat64())
		}
		weights[i-1] = w
		biases[i-1] = b
	}

	return &MLP{
		numLayers: len(sizes),
		weights:   weights,
		biases:    biases,
		config:    config,
	}
}

func (n *MLP) forward(x *mat.Dense) (as, zs []*mat.Dense) {
	as = make([]*mat.Dense, 0, len(n.weights)+1)
	zs = make([]*mat.Dense, 0, len(n.weights))
	as = append(as, x)
	cur := x

	for i := range n.weights {
		w := n.weights[i]
		b := n.biases[i]
		dot := new(mat.Dense)
		dot.Mul(cur, w)
		z := new(mat.Dense)
		z.Apply(addBias(b), dot)
		a := new(mat.Dense)
		a.Apply(applySigmoid, z)
		zs = append(zs, z)
		as = append(as, a)
		cur = a
	}
	return
}

func (n *MLP) backwards(x, y *mat.Dense) {
	as, _ := n.forward(x)
	batchSize, _ := x.Dims()

	outA := as[len(as)-1]
	err := new(mat.Dense)
	err.Sub(outA, y)

	spOut := new(mat.Dense)
	spOut.Apply(applySigmoidPrime, outA)

	delta := new(mat.Dense)
	delta.MulElem(err, spOut)

	nws := make([]*mat.Dense, len(n.weights))
	nbs := make([]*mat.Dense, len(n.biases))

	prevA := as[len(as)-2]
	gradW := new(mat.Dense)
	prevAT := mat.T(prevA)
	gradW.Mul(prevAT, delta)
	nws[len(n.weights)-1] = gradW

	rows, cols := delta.Dims()
	sumB := make([]float64, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			sumB[j] += delta.At(i, j)
		}
	}
	nbs[len(n.biases)-1] = mat.NewDense(1, cols, sumB)

	for layer := n.numLayers - 2; layer > 0; layer-- {
		aPrev := as[layer-1]

		wNext := n.weights[layer]
		wNextT := mat.T(wNext)
		wDelta := new(mat.Dense)
		wDelta.Mul(delta, wNextT)

		sp := new(mat.Dense)
		aCurr := as[layer]
		sp.Apply(applySigmoidPrime, aCurr)

		newDelta := new(mat.Dense)
		newDelta.MulElem(wDelta, sp)
		delta = newDelta

		prevAT := mat.T(aPrev)
		gradW := new(mat.Dense)
		gradW.Mul(prevAT, delta)
		nws[layer-1] = gradW

		r, c := delta.Dims()
		sumB := make([]float64, c)
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				sumB[j] += delta.At(i, j)
			}
		}
		nbs[layer-1] = mat.NewDense(1, c, sumB)
	}

	eta := n.config.Eta / float64(batchSize)
	for i := range n.weights {
		scaledW := new(mat.Dense)
		scaledW.Scale(eta, nws[i])
		wNew := new(mat.Dense)
		wNew.Sub(n.weights[i], scaledW)
		n.weights[i] = wNew

		scaledB := new(mat.Dense)
		scaledB.Scale(eta, nbs[i])
		bNew := new(mat.Dense)
		bNew.Sub(n.biases[i], scaledB)
		n.biases[i] = bNew
	}
}

func getRow(m *mat.Dense, row int) []float64 {
	_, cols := m.Dims()
	res := make([]float64, cols)
	for j := 0; j < cols; j++ {
		res[j] = m.At(row, j)
	}
	return res
}

func (n *MLP) Train(x, y *mat.Dense) {
	r, _ := x.Dims()
	b := n.config.BatchSize
	idx := make([]int, r)
	for i := range idx {
		idx[i] = i
	}

	start := time.Now()

	for e := 1; e <= n.config.Epochs; e++ {
		rand.Shuffle(len(idx), func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })

		for i := 0; i < r; i += b {
			k := i + b
			if k > r {
				k = r
			}
			colsX := x.Cols()
			colsY := y.Cols()
			xBatchData := make([]float64, (k-i)*colsX)
			yBatchData := make([]float64, (k-i)*colsY)

			for j := i; j < k; j++ {
				rowX := getRow(x, idx[j])
				rowY := getRow(y, idx[j])
				copy(xBatchData[(j-i)*colsX:], rowX)
				copy(yBatchData[(j-i)*colsY:], rowY)
			}
			
			xBatch := mat.NewDense(k-i, colsX, xBatchData)
			yBatch := mat.NewDense(k-i, colsY, yBatchData)

			n.backwards(xBatch, yBatch)
		}
	}

	fmt.Printf("Training completed in %s\n", time.Since(start))
}

func (n *MLP) Predict(x *mat.Dense) *mat.Dense {
	as, _ := n.forward(x)
	return as[len(as)-1]
}

func (n *MLP) Evaluate(x, y *mat.Dense) float64 {
	p := n.Predict(x)
	N, _ := p.Dims()
	var correct int
	for i := 0; i < N; i++ {
		ry := getRow(y, i)
		truth := oneHotDecode(ry)
		rp := getRow(p, i)
		if prediction(rp) == truth {
			correct++
		}
	}
	return float64(correct) / float64(N) * 100.0
}

func (n *MLP) TrainConcurrent(x, y *mat.Dense) {
	r, _ := x.Dims()
	b := n.config.BatchSize
	idx := make([]int, r)
	for i := range idx {
		idx[i] = i
	}

	start := time.Now()

	for e := 1; e <= n.config.Epochs; e++ {
		rand.Shuffle(len(idx), func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })

		var wg sync.WaitGroup
		var mu sync.Mutex

		for i := 0; i < r; i += b {
			end := i + b
			if end > r {
				end = r
			}
			wg.Add(1)
			go func(startIdx, endIdx int) {
				defer wg.Done()

				colsX := x.Cols()
				colsY := y.Cols()
				xs := make([]float64, (endIdx-startIdx)*colsX)
				ys := make([]float64, (endIdx-startIdx)*colsY)

				for j := startIdx; j < endIdx; j++ {
					rowX := getRow(x, idx[j])
					rowY := getRow(y, idx[j])
					copy(xs[(j-startIdx)*colsX:], rowX)
					copy(ys[(j-startIdx)*colsY:], rowY)
				}

				xBatch := mat.NewDense(endIdx-startIdx, colsX, xs)
				yBatch := mat.NewDense(endIdx-startIdx, colsY, ys)

				mu.Lock()
				n.backwards(xBatch, yBatch)
				mu.Unlock()
			}(i, end)
		}
		wg.Wait()
	}

	fmt.Printf("Concurrent training completed in %s\n", time.Since(start))
}
