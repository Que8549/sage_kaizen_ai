const canvas = document.getElementById('solarSystemCanvas');
const ctx = canvas.getContext('2d');

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// Function to draw a circle
function drawCircle(x, y, radius, color) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.closePath();
}

// Function to draw the planets
function drawPlanets() {
    const planets = [
        {name: "Mercury", radius: 2, color: "#E7B06B", distance: 50},
        {name: "Venus", radius: 4, color: "#E8C5B0", distance: 100},
        {name: "Earth", radius: 6, color: "#87CEEB", distance: 150},
        {name: "Mars", radius: 8, color: "#FFA07A", distance: 200},
        {name: "Jupiter", radius: 10, color: "#E0823A", distance: 250},
        {name: "Saturn", radius: 12, color: "#FFD700", distance: 300},
        {name: "Uranus", radius: 14, color: "#303F9F", distance: 350},
        {name: "Neptune", radius: 16, color: "#2196F3", distance: 400},
        {name: "Pluto", radius: 18, color: "#E91E63", distance: 450}
    ];

    planets.forEach(planet => {
        drawCircle(canvas.width / 2, canvas.height / 2, planet.radius, planet.color);
    });
}

// Draw the planets on the canvas
drawPlanets();