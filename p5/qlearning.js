// Q Table added
const xCoords = map.map((coord) => coord[0]);
const yCoords = map.map((coord) => coord[1]);

const maxX = Math.max(...xCoords);
const minX = Math.min(...xCoords);
const maxY = Math.max(...yCoords);
const minY = Math.min(...yCoords);

maxX = Math.round(maxX / 10);
minX = Math.round(minX / 10);
minY = Math.round(minY / 10);
maxY = Math.round(maxY / 10);

let qTable = {};

for (let x = minX; x <= maxX; x++) {
  for (let y = minY; y <= maxY; y++) {
    const state = `${x},${y}`;
    qTable[state] = {};
    actions.forEach((action) => {
      qTable[state][action] = 0;
    });
  }
}
// Learning Rate added
let alpha = 0.1;
// Discount value added
let gamma = 0.5;
// Update the q table
let state = `${x},${y}`;
qTable[state][action] = QVal(state, action, reward);
// Get State
// Get Q value added
function QVal(state, action, reward, direction) {
  let oldQVal = qTable[state][action];
  const parts = state.split(",");
  const x = parseInt(parts[0], 10);
  const y = parseInt(parts[1], 10);
  let dx = Math.round(Math.cos(direction));
  let dy = Math.round(Math.sin(direction));
  let newState = `${x + dx},${y + dy}`;
  let newQVal =
    oldQVal +
    alpha *
      (reward + gamma * Math.max(...Object.values(qTable[newState])) - oldQVal);
  return newQVal;
}
