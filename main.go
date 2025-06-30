package main

import (
	"fmt"
	"linearregression/lr"
	"linearregression/rd"
	"linearregression/utils"
	"log"
	"time"
)

func runMultivariateRegression(trainData, testData []rd.DataPoint, featureColumns []int) {
	trainX := make([][]float64, len(trainData))
	trainY := make([]float64, len(trainData))
	for i, dp := range trainData {
		trainX[i] = make([]float64, len(dp.Features))
		copy(trainX[i], dp.Features)
		trainY[i] = dp.Target
	}

	testX := make([][]float64, len(testData))
	testY := make([]float64, len(testData))
	for i, dp := range testData {
		testX[i] = make([]float64, len(dp.Features))
		copy(testX[i], dp.Features)
		testY[i] = dp.Target
	}

	model := lr.New(len(featureColumns))

	epochs := 1500
	learningRate := 0.003
	workers := 4

	fmt.Printf("Entrenando modelo multivariante con %d características\n", len(featureColumns))
	fmt.Printf("Parámetros de entrenamiento: épocas=%d, lr=%.4f, hilos=%d\n", epochs, learningRate, workers)

	startTime := time.Now()
	err := model.Fit(trainX, trainY, epochs, learningRate, workers)
	if err != nil {
		log.Printf("Error al entrenar el modelo multivariante: %v", err)
		return
	}
	trainingTime := time.Since(startTime)

	loss, converged := model.GetTrainingMetrics()
	weights := model.GetWeights()
	bias := model.GetBias()

	fmt.Printf("Entrenamiento completado en: %v\n", trainingTime)
	fmt.Printf("¿Convergió?: %v\n", converged)
	if len(loss) > 0 {
		fmt.Printf("Pérdida final de entrenamiento: %.6f\n", loss[len(loss)-1])
	}

	fmt.Printf("Parámetros finales del modelo:\n")

	for i, weight := range weights {
		fmt.Printf("- Característica_%d: %.6f\n", i, weight)
	}
	fmt.Printf("- Sesgo: %.6f\n", bias)

	r2Train, mseTrain, rmseTrain := model.Evaluate(trainX, trainY)
	r2Test, mseTest, rmseTest := model.Evaluate(testX, testY)

	fmt.Printf("Desempeño:\n")
	fmt.Printf("  Entrenamiento - R²: %.4f, MSE: %.4f, RMSE: %.4f\n", r2Train, mseTrain, rmseTrain)
	fmt.Printf("  Prueba        - R²: %.4f, MSE: %.4f, RMSE: %.4f\n", r2Test, mseTest, rmseTest)

	fmt.Printf("Predicciones de ejemplo:\n")
	fmt.Printf("%-8s %-12s %-8s\n", "Real", "Predicho", "Error")
	for i := 0; i < utils.Min(len(testX), 8); i++ {
		prediction := model.Predict(testX[i])
		error := prediction - testY[i]
		fmt.Printf("%-8.2f %-12.2f %-8.2f\n", testY[i], prediction, error)
	}
}


func main() {
	fmt.Println("=== Regresión Lineal con Go ===")
	fmt.Println("Cargando datos desde archivos CSV...")

	targetColumn := 6

	featureColumns := []int{}
	for i := 0; i < 50; i++ {
		if i != 6 {
			featureColumns = append(featureColumns, i)
		}
	}

	dateColumns := []int{1}

	trainFile := "Data/train_data.csv"
	trainData, err := rd.ReadCSV(trainFile, targetColumn, featureColumns, dateColumns)
	if err != nil {
		log.Fatalf("Error al cargar los datos de entrenamiento: %v", err)
	}

	fmt.Printf("\n=== Datos de Entrenamiento ===\n")
	rd.PrintDataSummary(trainData, "Características Académicas Seleccionadas")

	testFile := "Data/test_data.csv"
	testData, err := rd.ReadCSV(testFile, targetColumn, featureColumns, dateColumns)
	if err != nil {
		log.Fatalf("Error al cargar los datos de prueba: %v", err)
	}

	fmt.Printf("\n=== Datos de Prueba ===\n")
	rd.PrintDataSummary(testData, "Características Académicas Seleccionadas")

	fmt.Printf("\n=== Ejecutando Diferentes Modelos de Regresión ===\n")

	fmt.Printf("\n--- Regresión Multivariante ---\n")
	runMultivariateRegression(trainData, testData, featureColumns)

	fmt.Printf("\n=== ¡Programa completado exitosamente! ===\n")
}


