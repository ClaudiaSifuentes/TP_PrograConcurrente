package mymat

import (
	"fmt"
)

// Dense is a basic dense matrix implementation.
type Dense struct {
	rows, cols int
	data        []float64
}

// NewDense creates a new rows x cols matrix, using a backing slice.
func NewDense(rows, cols int, data []float64) *Dense {
	if data == nil {
		data = make([]float64, rows*cols)
	}
	if len(data) != rows*cols {
		panic(fmt.Sprintf("data length %d does not match %dx%d", len(data), rows, cols))
	}
	return &Dense{rows: rows, cols: cols, data: data}
}

// T returns the transpose of the matrix.
func T(m *Dense) *Dense{
	r, c := m.Dims()
	t := NewDense(c, r, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			t.Set(j, i, m.At(i, j))
		}
	}
	return t
}

// Dims returns matrix dimensions.
func (m *Dense) Dims() (r, c int) {
	return m.rows, m.cols
}

// At returns element at row i, col j (0-based).
func (m *Dense) At(i, j int) float64 {
	return m.data[i*m.cols+j]
}

// Set sets element at row i, col j.
func (m *Dense) Set(i, j int, v float64) {
	m.data[i*m.cols+j] = v
}


// Mul performs matrix multiplication: result = a * b.
func (m *Dense) Mul(a, b *Dense) *Dense {
    r1, c1 := a.Dims()
    r2, c2 := b.Dims()

    // Check for dimension mismatch
    if c1 != r2 {
        panic(fmt.Sprintf("mismatched dimensions for Mul: %dx%d and %dx%d", r1, c1, r2, c2))
    }

    // Create a new result matrix with appropriate dimensions
    res := NewDense(r1, c2, nil)

    // Perform matrix multiplication
    for i := 0; i < r1; i++ {
        for j := 0; j < c2; j++ {
            sum := 0.0
            for k := 0; k < c1; k++ {
                sum += a.At(i, k) * b.At(k, j)
            }
            res.Set(i, j, sum)
        }
		
    }
	m.rows, m.cols, m.data = res.rows, res.cols, res.data
    return res
}

// MulElem does element-wise multiplication: res_{ij} = a_{ij} * b_{ij}.
func (m *Dense) MulElem(a, b *Dense) *Dense {
	r, c := a.Dims()
	res := NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			res.Set(i, j, a.At(i, j)*b.At(i, j))
		}
	}
	m.rows, m.cols, m.data = res.rows, res.cols, res.data
	return res
}

// Scale scales matrix a by alpha: res = alpha * a.
func (m *Dense) Scale(alpha float64, a *Dense) *Dense {
	r, c := a.Dims()
	res := NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			res.Set(i, j, alpha*a.At(i, j))
		}
	}
	m.rows, m.cols, m.data = res.rows, res.cols, res.data
	return res
}

// Apply applies function f to each element: res_{ij} = f(i,j,v).
func (m *Dense) Apply(f func(i, j int, v float64) float64, a *Dense) *Dense {
	r, c := a.Dims()
	res := NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			res.Set(i, j, f(i, j, a.At(i, j)))
		}
	}
	m.rows, m.cols, m.data = res.rows, res.cols, res.data
	return res
}

func (m *Dense) RawMatrix() (rows, cols int, data []float64) {
	return m.rows, m.cols, m.data
}

// Rows returns the number of rows in the matrix.
func (m *Dense) Rows() int {
	return m.rows
}

// Cols returns the number of columns in the matrix.
func (m *Dense) Cols() int {
	return m.cols
}

// Sub subtracts two matrices: res = a - b.
func (m *Dense) Sub(a, b *Dense) *Dense {
	r1, c1 := a.Dims()
	r2, c2 := b.Dims()
	if r1 != r2 || c1 != c2 {
		panic("mismatched dimensions for Sub")
	}
	res := NewDense(r1, c1, nil)
	for i := 0; i < r1; i++ {
		for j := 0; j < c1; j++ {
			res.Set(i, j, a.At(i, j)-b.At(i, j))
		}
	}
	m.rows, m.cols, m.data = res.rows, res.cols, res.data
	return res
}