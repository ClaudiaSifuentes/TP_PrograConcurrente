/* algoritmo.pml - Modelo limpio y verificable */

byte updating = 0;
bool mutex = false;

active [2] proctype BackwardProcess() {
    int id = _pid + 1;
    
    do
    :: true ->
        // Secci┬ón no cr┬ítica
        printf("P%d: Pre-secci┬ón\n", id);
        
        // Adquirir mutex
        atomic {
            !mutex -> mutex = true;
            updating++;
            assert(updating == 1);
        }
        
        // Secci┬ón cr┬ítica
        printf("P%d: **CRITICAL**\n", id);
        updating--;
        mutex = false;
        
        // Evitar estado -end- no alcanzable
        if
        :: true -> skip;
        fi
    od
}

// Verificaci┬ón LTL directa
ltl me { [] (updating <= 1) }
