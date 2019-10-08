library(bnlearn)
d <- read.table('G:/Mi unidad/Desarrollo/datasets/erc1.csv', header = TRUE, sep = ',', dec = '.')
# construccion manual de la red
varnames <- c("Enfermedades", "dolorCabeza", "problemasRespiratorios")
ag <- empty.graph(varnames)
arcs(ag, ignore.cycles = T) <- data.frame(
  "from" = c("Enfermedades", "Enfermedades")
  "to" = c("dolorCabeza", "problemasRespiratorios"))
ag <- set.arc(ag, "Enfermedades", "dolorCabeza")
ag <- set.arc(ag, "Enfermedades", "problemasRespiratorios")
graph <- plot(ag)
modeloBN <- bn.fit(ag, d)
setTest1 <- read.table('C:/Users/Liber Proyectos/Documents/ALF-100/datos/gripe_testSet2.csv', header = TRUE, sep = ',', dec = '.')
cat("P(tener gripe_a | tiene dolor cabeza y problemas respiratorios) =", cpquery(modeloBN, (Enfermedades == "Gripe_A"), (problemasRespiratorios == "SI" & dolorCabeza == "SI")), "\n")
cat("P(tener gripe_comun | tiene dolor cabeza y no tiene problemas respiratorios) =", cpquery(modeloBN, (Enfermedades == "Gripe_Comun"), (problemasRespiratorios == "NO" & dolorCabeza == "SI")), "\n")

# investigar
prediccion <- predict(modeloBN, "Enfermedades", setTest1, method = "bayes-lw")

# construccion automatica por el algoritmo Grow-Shrink
t <- gs(d)
plot(t)
t2 <- iamb(d)
plot(t2)
all.equal(t, t2)
ntest(t)
ntest(t2)

blacklist <- data.frame(from = c("varY"), to = c("varX"))
t3 <- gs(d, blacklist = blacklist)
plot(t3)

modeloBN <- bn.fit(ag, d)
