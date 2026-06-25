/* ============================================================================
   VALIDACION_SSMS.sql  —  Factibilidad de pronóstico supervisado con Eqpcare.Fault
   ----------------------------------------------------------------------------
   Objetivo: confirmar que existe SEÑAL APRENDIBLE para predecir fallas de motor
   a partir del análisis de aceite. Cada bloque responde una pregunta concreta;
   corre uno por uno en SSMS y pega el resultado.

   Tablas:
     Eqpcare.Fault          -> eventos de falla (MiningEquipmentId, DateFrom, SmrFrom, Code, Description)
     Oil.LaboratoryData     -> muestras de aceite (MiningEquipmentId, Compartimiento, FechaMuestreo, Horometro, metales)
     Mine.MiningEquipment   -> equipo -> proyecto/modelo

   Join clave: Fault.MiningEquipmentId = Oil.LaboratoryData.MiningEquipmentId (ambos UNIQUEIDENTIFIER).
   Solo SELECT (lectura). WITH (NOLOCK) por cortesía en réplica de producción.
   ============================================================================ */


/* ---------------------------------------------------------------------------
   BLOQUE 1 — Panorama de Eqpcare.Fault
   Qué buscamos: ¿hay suficientes fallas, con equipo y fecha/Smr poblados?
   Decide: si total es bajo (cientos), el supervisado será frágil.
   --------------------------------------------------------------------------- */
SELECT
    COUNT(*)                                                  AS total_fallas,
    COUNT(DISTINCT MiningEquipmentId)                         AS equipos_con_falla,
    SUM(CASE WHEN MiningEquipmentId IS NULL THEN 1 ELSE 0 END) AS sin_equipo,
    SUM(CASE WHEN DateFrom IS NULL THEN 1 ELSE 0 END)         AS sin_fecha,
    SUM(CASE WHEN SmrFrom  IS NULL THEN 1 ELSE 0 END)         AS sin_smr,
    MIN(DateFrom)                                             AS fecha_min,
    MAX(DateFrom)                                             AS fecha_max
FROM Eqpcare.Fault WITH (NOLOCK);


/* ---------------------------------------------------------------------------
   BLOQUE 2 — Taxonomía de Code (qué tipos de falla existen)
   Qué buscamos: identificar qué codes son de MOTOR / aceite.
   Decide: con esto se llena faults.engine_codes en config.yaml.
   --------------------------------------------------------------------------- */
SELECT TOP 50
    Code,
    COUNT(*)            AS n,
    MIN(Description)    AS ejemplo_descripcion
FROM Eqpcare.Fault WITH (NOLOCK)
GROUP BY Code
ORDER BY n DESC;


/* ---------------------------------------------------------------------------
   BLOQUE 2b — Fallas cuya Description sugiere motor/aceite/tribología
   Qué buscamos: confirmar vocabulario real (afinar engine_keywords en config).
   --------------------------------------------------------------------------- */
SELECT TOP 100
    Code,
    Description,
    COUNT(*) AS n
FROM Eqpcare.Fault WITH (NOLOCK)
WHERE Description LIKE '%MOTOR%'   OR Description LIKE '%ACEITE%'
   OR Description LIKE '%CULATA%'  OR Description LIKE '%BIELA%'
   OR Description LIKE '%CIGUE%'   OR Description LIKE '%PISTON%'
   OR Description LIKE '%CAMISA%'  OR Description LIKE '%COJINETE%'
   OR Description LIKE '%TURBO%'   OR Description LIKE '%INYEC%'
   OR Description LIKE '%LUBRIC%'  OR Description LIKE '%REFRIGER%'
   OR Description LIKE '%BOMBA DE ACEITE%'
GROUP BY Code, Description
ORDER BY n DESC;


/* ---------------------------------------------------------------------------
   BLOQUE 3 — Solapamiento equipos: ¿las fallas ocurren en motores con aceite?
   Qué buscamos: cuántos equipos están en AMBAS tablas (sin esto, no hay datos).
   --------------------------------------------------------------------------- */
SELECT
  (SELECT COUNT(DISTINCT MiningEquipmentId)
     FROM Eqpcare.Fault WITH (NOLOCK)
     WHERE MiningEquipmentId IS NOT NULL)                       AS equipos_con_falla,
  (SELECT COUNT(DISTINCT MiningEquipmentId)
     FROM Oil.LaboratoryData WITH (NOLOCK)
     WHERE Compartimiento = 'MOTOR')                            AS equipos_aceite_motor,
  (SELECT COUNT(*) FROM (
       SELECT DISTINCT f.MiningEquipmentId
       FROM Eqpcare.Fault f WITH (NOLOCK)
       JOIN Oil.LaboratoryData o WITH (NOLOCK)
         ON o.MiningEquipmentId = f.MiningEquipmentId
        AND o.Compartimiento = 'MOTOR'
   ) t)                                                         AS equipos_en_ambos;


/* ---------------------------------------------------------------------------
   BLOQUE 3b — Fallas por proyecto/modelo (contexto de flota)
   Qué buscamos: si las fallas se concentran en ciertos proyectos/modelos.
   --------------------------------------------------------------------------- */
SELECT
    mp.Name              AS proyecto,
    ef.Model             AS modelo,
    COUNT(*)             AS fallas,
    COUNT(DISTINCT f.MiningEquipmentId) AS equipos
FROM Eqpcare.Fault f WITH (NOLOCK)
JOIN Mine.MiningEquipment me  WITH (NOLOCK) ON f.MiningEquipmentId = me.Id
LEFT JOIN Mine.MiningProject mp WITH (NOLOCK) ON me.MiningProjectId = mp.Id
LEFT JOIN Mine.EquipmentFleet ef WITH (NOLOCK) ON me.EquipmentFleetId = ef.Id
GROUP BY mp.Name, ef.Model
ORDER BY fallas DESC;


/* ---------------------------------------------------------------------------
   BLOQUE 4 — Panorama de muestras de aceite de MOTOR
   Qué buscamos: volumen, rango de fecha y de horómetro (para horizonte en horas).
   --------------------------------------------------------------------------- */
SELECT
    COUNT(*)                                              AS muestras_motor,
    COUNT(DISTINCT MiningEquipmentId)                     AS motores,
    MIN(FechaMuestreo)                                    AS fecha_min,
    MAX(FechaMuestreo)                                    AS fecha_max,
    SUM(CASE WHEN Horometro IS NULL THEN 1 ELSE 0 END)    AS sin_horometro,
    SUM(CASE WHEN FechaMuestreo IS NULL THEN 1 ELSE 0 END) AS sin_fecha
FROM Oil.LaboratoryData WITH (NOLOCK)
WHERE Compartimiento = 'MOTOR';


/* ---------------------------------------------------------------------------
   BLOQUE 5 — *** EL CRUCIAL *** Alineación temporal falla <- aceite previo
   Qué buscamos: de cada falla, ¿hay muestras de aceite en los 90/180 días previos?
   Decide: si "con_aceite_90d" es alto, hay historial para aprender el pre-falla.
            si es ~0, el aceite no precede a las fallas y NO hay pronóstico posible.
   (Para restringir a fallas de motor, descomenta el AND con los codes del Bloque 2.)
   --------------------------------------------------------------------------- */
WITH f AS (
    SELECT Id, MiningEquipmentId, DateFrom
    FROM Eqpcare.Fault WITH (NOLOCK)
    WHERE MiningEquipmentId IS NOT NULL
      AND DateFrom IS NOT NULL
      -- AND Code IN ('XXX','YYY')   -- <- codes de motor del Bloque 2
)
SELECT
    COUNT(*)                                      AS fallas_evaluadas,
    SUM(CASE WHEN x.c90  > 0 THEN 1 ELSE 0 END)   AS con_aceite_90d,
    SUM(CASE WHEN x.c180 > 0 THEN 1 ELSE 0 END)   AS con_aceite_180d,
    CAST(AVG(x.c180 * 1.0) AS DECIMAL(6,2))       AS prom_muestras_180d
FROM f
-- El DATEDIFF (usa f.DateFrom, externa) se calcula en la subconsulta derivada 'o';
-- el agregado externo solo referencia la columna interna o.dd. Así se evita el
-- error 8124 (agregado que mezcla columna interna con referencia externa).
CROSS APPLY (
    SELECT
      COUNT(CASE WHEN o.dd <= 90  THEN 1 END) AS c90,
      COUNT(CASE WHEN o.dd <= 180 THEN 1 END) AS c180
    FROM (
        SELECT DATEDIFF(day, oo.FechaMuestreo, f.DateFrom) AS dd
        FROM Oil.LaboratoryData oo WITH (NOLOCK)
        WHERE oo.MiningEquipmentId = f.MiningEquipmentId
          AND oo.Compartimiento = 'MOTOR'
          AND oo.FechaMuestreo <= f.DateFrom
          AND oo.FechaMuestreo >  DATEADD(day, -180, f.DateFrom)
    ) o
) x;


/* ---------------------------------------------------------------------------
   BLOQUE 6 — Tasa base de positivos (desbalance de clases)
   Qué buscamos: % de muestras de aceite que tienen una falla en los próximos 90d.
   Decide: si es <5%, hay desbalance fuerte -> usar PR-AUC y pesar la clase positiva.
   --------------------------------------------------------------------------- */
WITH o AS (
    SELECT LaboratoryDataId, MiningEquipmentId, FechaMuestreo
    FROM Oil.LaboratoryData WITH (NOLOCK)
    WHERE Compartimiento = 'MOTOR' AND FechaMuestreo IS NOT NULL
)
SELECT
    COUNT(*)                                                  AS muestras,
    SUM(CASE WHEN y.nf > 0 THEN 1 ELSE 0 END)                 AS positivas_90d,
    CAST(100.0 * SUM(CASE WHEN y.nf > 0 THEN 1 ELSE 0 END) / COUNT(*) AS DECIMAL(5,2)) AS pct_positivas
FROM o
CROSS APPLY (
    SELECT COUNT(*) AS nf
    FROM Eqpcare.Fault f WITH (NOLOCK)
    WHERE f.MiningEquipmentId = o.MiningEquipmentId
      AND f.DateFrom >  o.FechaMuestreo
      AND f.DateFrom <= DATEADD(day, 90, o.FechaMuestreo)
) y;


/* ---------------------------------------------------------------------------
   BLOQUE 7 — Lead time: días entre la última muestra previa y la falla
   Qué buscamos: ¿con cuánta anticipación "ve" el aceite la falla?
   Decide: define un horizonte realista (si el gap medio es ~30d, H=90d es sano).
   --------------------------------------------------------------------------- */
;WITH f AS (
    SELECT Id, MiningEquipmentId, DateFrom
    FROM Eqpcare.Fault WITH (NOLOCK)
    WHERE MiningEquipmentId IS NOT NULL AND DateFrom IS NOT NULL
)
SELECT
    COUNT(*)            AS fallas_con_muestra_previa,
    AVG(g.gap)          AS gap_prom_dias,
    MIN(g.gap)          AS gap_min_dias,
    MAX(g.gap)          AS gap_max_dias
FROM f
CROSS APPLY (
    SELECT TOP 1 DATEDIFF(day, o.FechaMuestreo, f.DateFrom) AS gap
    FROM Oil.LaboratoryData o WITH (NOLOCK)
    WHERE o.MiningEquipmentId = f.MiningEquipmentId
      AND o.Compartimiento = 'MOTOR'
      AND o.FechaMuestreo <= f.DateFrom
    ORDER BY o.FechaMuestreo DESC
) g;


/* ---------------------------------------------------------------------------
   BLOQUE 8 — (Opcional) Horizonte en HORAS de operación (Smr) en vez de días
   Qué buscamos: ¿Fault.SmrFrom y Oil.Horometro están poblados para usar horas?
   Decide: si ambos > ~90% poblados, preferimos horizonte en horas (200-500 h).
   --------------------------------------------------------------------------- */
SELECT
    (SELECT CAST(100.0*SUM(CASE WHEN SmrFrom IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*) AS DECIMAL(5,2))
       FROM Eqpcare.Fault WITH (NOLOCK))                                   AS pct_fault_con_smr,
    (SELECT CAST(100.0*SUM(CASE WHEN Horometro IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*) AS DECIMAL(5,2))
       FROM Oil.LaboratoryData WITH (NOLOCK) WHERE Compartimiento='MOTOR') AS pct_aceite_con_horometro;


/* ===========================================================================
   FASE 2 — Objetivo correcto: trayectoria de CONDICION del laboratorio
   ---------------------------------------------------------------------------
   Conclusión de Fase 1: Eqpcare.Fault es un LOG DE TELEMETRÍA (1.9M eventos,
   ~15k por equipo, códigos de alarma operativa), no fallas de mantenimiento.
   La verdad de campo útil es el veredicto del propio laboratorio (Condicion /
   Estado) en Oil.LaboratoryData. Estos bloques lo caracterizan para definir la
   etiqueta supervisada: "¿el motor entra en condición adversa en (t, t+H]?".
   =========================================================================== */


/* ---------------------------------------------------------------------------
   BLOQUE 9 — Valores reales de Condicion y Estado (poblamiento y vocabulario)
   Decide: cuál columna usar (la mejor poblada) y qué valores son "adversos".
   --------------------------------------------------------------------------- */
SELECT 'Condicion' AS col, ISNULL(Condicion,'(NULL)') AS valor, COUNT(*) AS n
FROM Oil.LaboratoryData WITH (NOLOCK) WHERE Compartimiento='MOTOR'
GROUP BY Condicion
UNION ALL
SELECT 'Estado' AS col, ISNULL(Estado,'(NULL)') AS valor, COUNT(*) AS n
FROM Oil.LaboratoryData WITH (NOLOCK) WHERE Compartimiento='MOTOR'
GROUP BY Estado
ORDER BY col, n DESC;


/* ---------------------------------------------------------------------------
   BLOQUE 10 — Longitud de serie por motor (¿hay suficientes para ventanas?)
   Decide: cuántos motores tienen >= window_size+horizon muestras (entrenables).
   --------------------------------------------------------------------------- */
;WITH c AS (
    SELECT MiningEquipmentId, COUNT(*) AS n
    FROM Oil.LaboratoryData WITH (NOLOCK)
    WHERE Compartimiento='MOTOR' AND FechaMuestreo IS NOT NULL
    GROUP BY MiningEquipmentId
)
SELECT
    COUNT(*)                                       AS motores,
    SUM(CASE WHEN n >= 11 THEN 1 ELSE 0 END)       AS con_11omas,
    SUM(CASE WHEN n >= 20 THEN 1 ELSE 0 END)       AS con_20omas,
    CAST(AVG(n*1.0) AS DECIMAL(6,1))               AS prom_muestras,
    MIN(n) AS minimo, MAX(n) AS maximo
FROM c;


/* ---------------------------------------------------------------------------
   BLOQUE 11 — *** CLAVE *** Matriz de transición Condicion(t) -> Condicion(t+1)
   Qué buscamos: tasa base del objetivo y si la condición es "leading".
   Lee las filas: cuántas muestras pasan de su condición actual a la siguiente.
   Decide: tasa base de "siguiente = adversa" (desbalance) y horizonte realista.
   (Cambia Condicion por Estado si el Bloque 9 muestra que Estado está mejor poblado.)
   --------------------------------------------------------------------------- */
;WITH s AS (
    SELECT MiningEquipmentId, FechaMuestreo,
           UPPER(LTRIM(RTRIM(Condicion))) AS cond_now,
           LEAD(UPPER(LTRIM(RTRIM(Condicion))))
               OVER (PARTITION BY MiningEquipmentId ORDER BY FechaMuestreo) AS cond_next
    FROM Oil.LaboratoryData WITH (NOLOCK)
    WHERE Compartimiento='MOTOR' AND FechaMuestreo IS NOT NULL
)
SELECT ISNULL(cond_now,'(NULL)') AS cond_now,
       ISNULL(cond_next,'(NULL)') AS cond_next,
       COUNT(*) AS n
FROM s
GROUP BY cond_now, cond_next
ORDER BY n DESC;


/* ---------------------------------------------------------------------------
   BLOQUE 12 — (v2, opcional) Códigos de Fault candidatos a falla REAL de motor
   Qué buscamos: códigos de motor/lubricación POCO frecuentes (rareza ~ severidad)
   y en cuántos equipos aparecen. Solo para enriquecer/validar en v2, no para v1.
   --------------------------------------------------------------------------- */
SELECT Code, MIN(Description) AS descripcion,
       COUNT(*) AS n, COUNT(DISTINCT MiningEquipmentId) AS equipos
FROM Eqpcare.Fault WITH (NOLOCK)
WHERE Description LIKE '%LUBRIC%'   OR Description LIKE '%ACEITE%'
   OR Description LIKE '%OIL%'      OR Description LIKE '%CRANK%'
   OR Description LIKE '%CYLINDER%' OR Description LIKE '%BEARING%'
   OR Description LIKE '%COOLANT%'  OR Description LIKE '%OVERHEAT%'
   OR Description LIKE '%LOW.*PRES%'
GROUP BY Code
HAVING COUNT(*) BETWEEN 1 AND 2000
ORDER BY n ASC;


/* ===========================================================================
   FASE 3 — Cerrar la semántica y el VOLUMEN de la etiqueta supervisada v1
   ---------------------------------------------------------------------------
   Escala de severidad unificada (0=Normal, 1=Monitoreo, 2=Precaución, 3=Crítico)
   coalesce(Condicion, Estado). Estos bloques confirman: (B13) cómo se relacionan
   Condicion y Estado, (B14) cuántas muestras/motores quedan etiquetables, y
   (B15) cuántos POSITIVOS hay en el horizonte -> decide LSTM vs gradient boosting.
   =========================================================================== */


/* ---------------------------------------------------------------------------
   BLOQUE 13 — Relación Condicion x Estado (validar el mapeo de severidad)
   Decide: si Condicion '2' coincide con PRECAUTORIO y '3' con CRÍTICO, etc.
   --------------------------------------------------------------------------- */
SELECT ISNULL(Condicion,'(NULL)') AS condicion,
       ISNULL(UPPER(LTRIM(RTRIM(Estado))),'(NULL)') AS estado,
       COUNT(*) AS n
FROM Oil.LaboratoryData WITH (NOLOCK)
WHERE Compartimiento='MOTOR'
GROUP BY Condicion, UPPER(LTRIM(RTRIM(Estado)))
ORDER BY n DESC;


/* ---------------------------------------------------------------------------
   BLOQUE 14 — Volumen etiquetable con la severidad unificada
   Decide: cuántas muestras codificadas y cuántos motores con >=11 codificadas.
   --------------------------------------------------------------------------- */
;WITH base AS (
    SELECT MiningEquipmentId,
      CASE
        WHEN Condicion = '3' THEN 3
        WHEN Condicion = '2' THEN 2
        WHEN Condicion = '1' THEN 0
        WHEN UPPER(LTRIM(RTRIM(Estado))) IN ('CRÍTICO','CRITICO','C') THEN 3
        WHEN UPPER(LTRIM(RTRIM(Estado))) IN ('PRECAUTORIO','P')       THEN 2
        WHEN UPPER(LTRIM(RTRIM(Estado))) IN ('MONITOREO')             THEN 1
        WHEN UPPER(LTRIM(RTRIM(Estado))) IN ('NORMAL','N')            THEN 0
        ELSE NULL
      END AS sev
    FROM Oil.LaboratoryData WITH (NOLOCK)
    WHERE Compartimiento='MOTOR' AND FechaMuestreo IS NOT NULL
),
coded AS (SELECT * FROM base WHERE sev IS NOT NULL)
SELECT
    (SELECT COUNT(*) FROM coded)                                   AS muestras_codificadas,
    (SELECT SUM(CASE WHEN sev>=2 THEN 1 ELSE 0 END) FROM coded)    AS sev_2omas,
    (SELECT SUM(CASE WHEN sev=3 THEN 1 ELSE 0 END) FROM coded)     AS criticas_3,
    (SELECT COUNT(DISTINCT MiningEquipmentId) FROM coded)          AS motores_codificados,
    (SELECT COUNT(*) FROM (SELECT MiningEquipmentId, COUNT(*) n FROM coded
                           GROUP BY MiningEquipmentId HAVING COUNT(*)>=11) t) AS motores_11omas_codif;


/* ---------------------------------------------------------------------------
   BLOQUE 15 — *** DECISIVO *** Positivos en el horizonte (próxima muestra codificada)
   Qué buscamos: con LEAD sobre muestras codificadas, cuántos pares tienen la
   siguiente severidad adversa dentro de 120/180 días.
   Decide: nº de POSITIVOS entrenables -> >~500 viable LSTM; si es bajo, usar GBT.
   --------------------------------------------------------------------------- */
;WITH base AS (
    SELECT MiningEquipmentId, FechaMuestreo,
      CASE
        WHEN Condicion = '3' THEN 3
        WHEN Condicion = '2' THEN 2
        WHEN Condicion = '1' THEN 0
        WHEN UPPER(LTRIM(RTRIM(Estado))) IN ('CRÍTICO','CRITICO','C') THEN 3
        WHEN UPPER(LTRIM(RTRIM(Estado))) IN ('PRECAUTORIO','P')       THEN 2
        WHEN UPPER(LTRIM(RTRIM(Estado))) IN ('MONITOREO')             THEN 1
        WHEN UPPER(LTRIM(RTRIM(Estado))) IN ('NORMAL','N')            THEN 0
        ELSE NULL
      END AS sev
    FROM Oil.LaboratoryData WITH (NOLOCK)
    WHERE Compartimiento='MOTOR' AND FechaMuestreo IS NOT NULL
),
coded AS (SELECT * FROM base WHERE sev IS NOT NULL),
lead AS (
    SELECT MiningEquipmentId, FechaMuestreo, sev,
        LEAD(sev) OVER (PARTITION BY MiningEquipmentId ORDER BY FechaMuestreo) AS sev_next,
        DATEDIFF(day, FechaMuestreo,
                 LEAD(FechaMuestreo) OVER (PARTITION BY MiningEquipmentId ORDER BY FechaMuestreo)) AS gap_next
    FROM coded
)
SELECT
    COUNT(*)                                                              AS pares_codificados,
    SUM(CASE WHEN sev_next>=2 AND gap_next<=120 THEN 1 ELSE 0 END)        AS pos_sev2_120d,
    SUM(CASE WHEN sev_next=3  AND gap_next<=120 THEN 1 ELSE 0 END)        AS pos_crit_120d,
    SUM(CASE WHEN sev_next>=2 AND gap_next<=180 THEN 1 ELSE 0 END)        AS pos_sev2_180d
FROM lead
WHERE sev_next IS NOT NULL;


/* ===========================================================================
   FASE 4 — Diagnóstico de NO-GENERALIZACIÓN (test ROC≈0.5, GBT<0.5)
   ---------------------------------------------------------------------------
   Hipótesis: Condicion (valores 1/2/3) y Estado (NORMAL/MONITOREO/...) son ERAS
   distintas. El split temporal entrena en una era y evalúa en otra -> la etiqueta
   cambia de significado y el modelo no transfiere. Este bloque lo confirma.
   =========================================================================== */


/* ---------------------------------------------------------------------------
   BLOQUE 16 — *** CLAVE *** Línea de tiempo de cada esquema de codificación
   Qué buscamos: por año, cuántas muestras usan Condicion vs Estado, y la tasa
   de adversas (sev>=2). Si Condicion domina años viejos y Estado años nuevos
   (o viceversa), el split temporal mezcla eras -> causa del fallo en test.
   Decide: si hay eras, entrenar/evaluar dentro de UNA era, o split por equipo.
   --------------------------------------------------------------------------- */
;WITH base AS (
    SELECT YEAR(FechaMuestreo) AS anio,
      CASE WHEN Condicion IN ('1','2','3') THEN 1 ELSE 0 END AS es_condicion,
      CASE WHEN UPPER(LTRIM(RTRIM(Estado))) IN
           ('NORMAL','N','MONITOREO','PRECAUTORIO','P','CRÍTICO','CRITICO','C')
           THEN 1 ELSE 0 END AS es_estado,
      CASE
        WHEN Condicion = '3' THEN 3
        WHEN Condicion = '2' THEN 2
        WHEN Condicion = '1' THEN 0
        WHEN UPPER(LTRIM(RTRIM(Estado))) IN ('CRÍTICO','CRITICO','C') THEN 3
        WHEN UPPER(LTRIM(RTRIM(Estado))) IN ('PRECAUTORIO','P')       THEN 2
        WHEN UPPER(LTRIM(RTRIM(Estado))) IN ('MONITOREO')             THEN 1
        WHEN UPPER(LTRIM(RTRIM(Estado))) IN ('NORMAL','N')            THEN 0
        ELSE NULL
      END AS sev
    FROM Oil.LaboratoryData WITH (NOLOCK)
    WHERE Compartimiento='MOTOR' AND FechaMuestreo IS NOT NULL
)
SELECT anio,
       COUNT(*)                                                  AS muestras,
       SUM(es_condicion)                                         AS cod_condicion,
       SUM(es_estado)                                            AS cod_estado,
       SUM(CASE WHEN sev IS NOT NULL THEN 1 ELSE 0 END)          AS codificadas,
       SUM(CASE WHEN sev>=2 THEN 1 ELSE 0 END)                   AS adversas,
       CAST(100.0*SUM(CASE WHEN sev>=2 THEN 1 ELSE 0 END)
            / NULLIF(SUM(CASE WHEN sev IS NOT NULL THEN 1 ELSE 0 END),0) AS DECIMAL(5,1)) AS pct_adversas
FROM base
GROUP BY anio
ORDER BY anio;
