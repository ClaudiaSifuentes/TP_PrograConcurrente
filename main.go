package main

import (
	"encoding/csv"
	"fmt"
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
		target = (target - 1) / 9.0
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
	train_x, train_y := loadDF("data/train_augmented.csv")
	test_x, test_y := loadDF("data/test.csv")

	config := nn.Config{
		Eta:       0.3,
		Epochs:    100,
		BatchSize: 100,
	}

	mlp := nn.NewMLP([]int{11, 32, 16, 8, 1}, config)

	mlp.TrainConcurrent(train_x, train_y)
	mae, mse := mlp.Evaluate(test_x, test_y)
	fmt.Println("MAE: ", mae)
	fmt.Println("MSE: ", mse)

}
