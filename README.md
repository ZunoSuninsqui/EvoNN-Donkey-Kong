# Donkey Kong IA medio evolutiva con Q-Learning

## Introducción muy nuestra
Hicimos un mini Donkey Kong en `pygame` porque queríamos jugar con un algoritmo evolutivo casero y ver si una red neuronal podía aprender a esquivar barriles. El proyecto suena grande, pero en realidad es un montón de rectángulos que se mueven y un agente curioso tomando decisiones. Se realizó el juego con pygame utilizando una imagen cómo del mapa para generar el mapa con bloques de colision que la IA utilizó para buscar glitches. También nos pasó que se propuso utilizar una imagen de background para que se viera más presentable pero generaba problemas con el entrenamiento ya que requería cargar la imagen para poder hacerlo y a veces no lo hacía, así que lo contamos porque duele.

## Cómo pensamos la IA y el toque evolutivo
El cerebro de la IA está en `donkey kong q learning.py`. Queríamos algo que sonara a red neuronal porque la consigna lo pedía, así que armamos una red chiquita con capas densas y la usamos con Q-Learning. Se utilizó Q-Learning siguiendo con la normativa de que fuera un red neuronal, con un detalle extra, todo esto fue inspiración del youtuber codeBullet que hace este tipo de contenido en youtube. No es un mega algoritmo genético, pero la exploración con `epsilon` decayendo y el buffer de repeticiones nos recordó a la evolución gradual: probar cosas, guardar lo que sirve y repetir.

## El juego en pygame y el mapa
El archivo principal arma la ventana, fija el tamaño y define los colores, todo sin comentarios para que el código quede limpio pero explicado aquí. Creamos un `Game` que arma plataformas, escaleras y zonas donde los barriles se voltean. La idea original era cargar un mapa desde una imagen y, en espíritu, seguimos pensando así aunque ahora las plataformas están en listas de `pygame.Rect`. El background se carga con `Game._load_background` usando la ruta `BACKGROUND_PATH`, pero cuando entrenamos ponemos `use_background=False` porque el entrenamiento se ponía quisquilloso. Se propuso utilizar una imagen de background para que se viera más presentable pero generaba problemas con el entrenamiento ya que requería cargar la imagen para poder hacerlo y a veces no lo hacía, y lo repetimos porque fue un dolor real.

## Glitches y cómo los buscamos
Con el mapa en rectángulos, el agente explora saltos y escaleras intentando llegar a la princesa. Las escaleras tienen tolerancias raras y los barriles rebotan en zonas invisibles (`turn_zones`), así que a veces aparecen movimientos que parecen glitches. Se realizó el juego con pygame utilizando una imagen cómo del mapa para generar el mapa con bloques de colision que la IA utilizó para buscar glitches, y aunque ahora los bloques vienen de números duros, la IA sigue intentando colarse por huecos y bordes porque los rects permiten trucos de pixel-perfect.

## Q-Learning y la pseudo red neuronal
La clase `NeuralNetwork` monta dos capas densas con ReLU y se guarda/carga desde JSON (`AI_WEIGHTS_PATH`). La clase `AIAgent` envuelve todo con un `ReplayBuffer` que almacena experiencias `(estado, acción, recompensa, siguiente_estado, done)`. Se utilizó Q-Learning siguiendo con la normativa de que fuera un red neuronal, con un detalle extra, todo esto fue inspiración del youtuber codeBullet que hace este tipo de contenido en youtube. Nos gustó repetir esto porque realmente vimos sus videos mientras codificábamos.

El estado que alimentamos tiene posiciones normalizadas del jugador, del barril más cercano y de la princesa, más flags como si está en escalera o en el suelo. El agente calcula recompensas simples: sobrevivir suma un poquito, acercarse a la princesa suma más, morir resta un montón y ganar da 100. Entrenamos cada pocos frames (`train_step`) con backprop manual y un `epsilon` que decae para pasar de exploración a explotación.

## Detalles importantes de la implementación
- Se tiene una velocidad constante de barriles para poder que el entrenamiento y lista de movimientos siempre sean precisos. Los barriles salen cada cierto número de frames y se dan la vuelta al chocar con `turn_zones`.
- El jugador (`Player`) alinea su `centerx` con la escalera cuando decide subir para evitar que se quede colgado. La gravedad y los saltos se controlan con constantes (`GRAVITY`, `JUMP_FORCE`) definidas al inicio.
- El menú tiene tres modos: jugar humano, entrenar IA (sin fondo para que corra más rápido) y ver la IA en demo. Cambiar la velocidad de entrenamiento con teclas 1/2/3 acelera el loop para que aprenda más rápido aunque se vea como fast-forward.
- Guardamos pesos cada 50 episodios y también al salir para no perder el progreso. Si existe `donkey_kong_ai_weights.json` se carga automáticamente.

## Cómo ejecutar y requisitos
1. Necesitas Python 3 y `pygame` + `numpy` instalados (`pip install pygame numpy`).
2. Ejecuta `python "donkey kong q learning.py"` desde la raíz del repo.
3. En el menú elige "JUGAR", "ENTRENAR IA" o "VER IA JUGANDO". En entrenamiento puedes presionar `1`, `2` o `3` para cambiar la velocidad y `ESC` para volver al menú guardando.

## Limitaciones y problemas conocidos
- La carga del background puede fallar si la ruta no existe; por eso el entrenamiento va sin imagen. Se propuso utilizar una imagen de background para que se viera más presentable pero generaba problemas con el entrenamiento ya que requería cargar la imagen para poder hacerlo y a veces no lo hacía, así que lo dejamos opcional.
- No hay detección precisa de hitboxes redondas; todo es cuadrado y eso genera glitches divertidos.
- No implementamos un verificador de convergencia, así que la IA puede quedarse dando vueltas si el `epsilon` no baja lo suficiente.

## Posibles mejoras
- Leer realmente un mapa desde una imagen para generar plataformas y escaleras en vez de escribir coordenadas a mano, manteniendo la idea de bloques de colisión automáticos.
- Afinar las recompensas para que valore más las rutas rápidas y penalice saltos inútiles.
- Añadir un modo de grabar repeticiones para estudiar los glitches que la IA encuentra.

## Conclusión humana
Aprendimos que mezclar pygame con Q-Learning es más trabajo de lo que parece, sobre todo cuando se quiere algo que suene a "red neuronal" pero funcione rápido. Nos divertimos viendo a la IA estrellarse contra barriles, y entendimos que mantener una velocidad constante de barriles ayuda a que el entrenamiento sea coherente. Se realizó el juego con pygame utilizando una imagen cómo del mapa para generar el mapa con bloques de colision que la IA utilizó para buscar glitches, y aunque el código quedó sin comentarios, acá tratamos de contar cada detalle. También repetimos que se utilizó Q-Learning siguiendo con la normativa de que fuera un red neuronal, con un detalle extra, todo esto fue inspiración del youtuber codeBullet que hace este tipo de contenido en youtube, porque realmente fue nuestra motivación.
