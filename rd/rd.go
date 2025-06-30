package rd

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"
)

type DataPoint struct {
	Features []float64
	Target   float64
}

func ReadCSV(filename string, targetColumn int, featureColumns []int, dateColumns []int) ([]DataPoint, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("error al abrir el archivo %s: %v", filename, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("error al leer el CSV: %v", err)
	}

	if len(records) < 2 {
		return nil, fmt.Errorf("El archivo CSV debe tener al menos encabezado y una fila de datos")
	}

	var dataPoints []DataPoint
	for i := 1; i < len(records); i++ {
		record := records[i]

		target, err := strconv.ParseFloat(record[targetColumn], 64)
		if err != nil {
			log.Printf("Advertencia: se omite la fila %d, valor objetivo inválido: %v", i, err)
			continue
		}

		features := make([]float64, len(featureColumns))
		validRow := true
		for j, colIdx := range featureColumns {
			if colIdx >= len(record) {
				log.Printf("Advertencia: se omite la fila %d, índice de columna %d fuera de rango", i, colIdx)
				validRow = false
				break
			}
			features[j], err = parseValueToFloat(record[colIdx], colIdx, dateColumns)
			if err != nil {
				log.Printf("Advertencia: se omite la fila %d, valor de característica inválido en la columna %d: %v", i, colIdx, err)
				validRow = false
				break
			}
		}

		if validRow {
			dataPoints = append(dataPoints, DataPoint{
				Features: features,
				Target:   target,
			})
		}
	}

	return dataPoints, nil
}

func ExtractXY(dataPoints []DataPoint, featureIndex int) ([]float64, []float64) {
	xs := make([]float64, len(dataPoints))
	ys := make([]float64, len(dataPoints))

	for i, dp := range dataPoints {
		if featureIndex < len(dp.Features) {
			xs[i] = dp.Features[featureIndex]
		}
		ys[i] = dp.Target
	}

	return xs, ys
}

func PrintDataSummary(dataPoints []DataPoint, featureName string) {
	if len(dataPoints) == 0 {
		fmt.Println("No se cargaron puntos de datos")
		return
	}

	fmt.Printf("Resumen de Datos:\n")
	fmt.Printf("- Número de muestras: %d\n", len(dataPoints))
	fmt.Printf("- Número de características: %d\n", len(dataPoints[0].Features))

	var sumTarget, minTarget, maxTarget float64
	minTarget = dataPoints[0].Target
	maxTarget = dataPoints[0].Target

	for _, dp := range dataPoints {
		sumTarget += dp.Target
		if dp.Target < minTarget {
			minTarget = dp.Target
		}
		if dp.Target > maxTarget {
			maxTarget = dp.Target
		}
	}

	avgTarget := sumTarget / float64(len(dataPoints))
	fmt.Printf("- Variable objetivo (%s):\n", "PROMEDIO_PONDERADO_DEL_PERIODO_ANTERIOR")
	fmt.Printf("  Mín: %.2f, Máx: %.2f, Prom: %.2f\n", minTarget, maxTarget, avgTarget)

	if len(dataPoints[0].Features) > 0 {
		var sumFeature, minFeature, maxFeature float64
		minFeature = dataPoints[0].Features[0]
		maxFeature = dataPoints[0].Features[0]

		for _, dp := range dataPoints {
			if len(dp.Features) > 0 {
				sumFeature += dp.Features[0]
				if dp.Features[0] < minFeature {
					minFeature = dp.Features[0]
				}
				if dp.Features[0] > maxFeature {
					maxFeature = dp.Features[0]
				}
			}
		}

		avgFeature := sumFeature / float64(len(dataPoints))
		fmt.Printf("- Característica (%s):\n", featureName)
		fmt.Printf("  Mín: %.2f, Máx: %.2f, Prom: %.2f\n", minFeature, maxFeature, avgFeature)
	}
}

func parseValueToFloat(value string, columnIndex int, dateColumns []int) (float64, error) {
	value = strings.TrimSpace(value)

	for _, dateCol := range dateColumns {
		if columnIndex == dateCol {
			return parseDateToFloat(value)
		}
	}

	if value == "True" || value == "true" {
		return 1.0, nil
	}
	if value == "False" || value == "false" {
		return 0.0, nil
	}

	return strconv.ParseFloat(value, 64)
}

func parseDateToFloat(dateStr string) (float64, error) {
	date, err := time.Parse("2006-01-02", dateStr)
	if err != nil {
		return 0, fmt.Errorf("formato de fecha inválido: %v", err)
	}

	epoch := time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC)
	daysSinceEpoch := date.Sub(epoch).Hours() / 24

	return daysSinceEpoch, nil
}
