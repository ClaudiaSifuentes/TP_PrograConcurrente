package main

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"

	"neural_network/modules/nn"

	mat "neural_network/modules/mymat"
)

func loadDF(path string) (*mat.Dense, *mat.Dense) {
	file, err := os.Open(path)
	if err != nil {
		log.Fatalf("failed to open file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = ';'

	records, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("failed to read CSV: %v", err)
	}

	if len(records) < 2 {
		log.Fatalf("CSV file does not contain enough rows")
	}

	records = records[1:] // Skip the header row

	rows := len(records)
	cols := len(records[0]) - 1 // Last column is the target

	xData := make([]float64, rows*cols)
	yData := make([]float64, rows)

	for i, record := range records {
		for j := 0; j < cols; j++ {
			val, err := strconv.ParseFloat(record[j], 64)
			if err != nil {
				log.Fatalf("failed to parse float: %v", err)
			}
			xData[i*cols+j] = val
		}
		target, err := strconv.ParseFloat(record[cols], 64)
		if err != nil {
			log.Fatalf("failed to parse target: %v", err)
		}
		yData[i] = target
	}

	x := mat.NewDense(rows, cols, xData)
	y := mat.NewDense(rows, 1, yData)

	return x, y
}

func main() {
	x, y := loadDF("train_augmented.csv")

	config := nn.Config{
		Eta:       0.1,
		Epochs:    100,
		BatchSize: 1000,
	}

	mlp1 := nn.NewMLP([]int{11, 16, 1}, config)

	mlp1.Train(x, y)

	mlp2 := nn.NewMLP([]int{11, 16, 1}, config)

	mlp2.TrainConcurrent(x, y)

}
