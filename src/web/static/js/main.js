// main.js

// 1. Inicializar Mapa (CartoDB Dark Matter para dar ese look Premium)
const map = L.map('map', {
    center: [40.7128, -74.0060], // New York
    zoom: 11,
    zoomControl: false // Lo ocultamos para que quede más visual
});

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
    subdomains: 'abcd',
    maxZoom: 20
}).addTo(map);

// 2. Traer y Renderizar Geometría (Desde nuestro backend con el CSV local)
let markersMap = {}; // Registra estacion ID -> Marker
const SUBWAY_STATIONS_URL = '/api/stations';

// Paleta oficial de colores MTA
const ROUTE_COLORS = {
    '1': '#EE352E', '2': '#EE352E', '3': '#EE352E', // Red
    '4': '#00933C', '5': '#00933C', '6': '#00933C', // Green
    '7': '#B933AD',                               // Purple
    'A': '#0039A6', 'C': '#0039A6', 'E': '#0039A6', // Blue
    'B': '#FF6319', 'D': '#FF6319', 'F': '#FF6319', 'M': '#FF6319', // Orange
    'G': '#6CBE45',                               // Light Green
    'J': '#996633', 'Z': '#996633',                // Brown
    'L': '#A7A9AC',                               // Gray
    'N': '#FCCC0A', 'Q': '#FCCC0A', 'R': '#FCCC0A', 'W': '#FCCC0A', // Yellow
    'S': '#808183', 'SIR': '#0039A6'               // Shuttle/SIR
};

function getStationColor(routes) {
    if (!routes) return '#3B82F6'; // Default accent color
    const primaryRoute = routes.split(' ')[0];
    return ROUTE_COLORS[primaryRoute] || '#3B82F6';
}

fetch(SUBWAY_STATIONS_URL)
    .then(res => res.json())
    .then(stations => {
        stations.forEach(station => {
            const color = getStationColor(station.routes);
            const markerValue = L.divIcon({
                className: 'station-marker',
                iconSize: [10, 10],
                html: `<div style="background-color: ${color}; width: 100%; height: 100%; border-radius: 50%; border: 1px solid white; box-shadow: 0 0 5px ${color}"></div>`
            });
            
            // Creamos el marker en Leaflet
            const marker = L.marker([station.lat, station.lon], {icon: markerValue});
            
            // Muestra popup nativo al hover
            marker.bindPopup(`
                <div class="popup-content">
                    <h3 style="color: ${color}">${station.name}</h3>
                    <p><strong>Rutas:</strong> ${station.routes}</p>
                    <span style="font-size: 0.7rem; color: #94A3B8">ID Estación: ${station.id}</span>
                </div>
            `);
            
            markersMap[station.id] = marker.addTo(map);
        });
        console.log(`${stations.length} estaciones instanciadas con sus colores de línea.`);
    })
    .catch(err => console.error("Error al cargar dataset de estaciones desde API:", err));

// 3. Conexión WebSocket para Predicciones periódicas de FastAPI
const wsUrl = `ws://${window.location.host}/ws/live-updates`;
const socket = new WebSocket(wsUrl);

socket.onopen = () => console.log("Conectado a FastAPI WebSocket Core");

socket.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    
    if (payload.type === 'update' || payload.type === 'initial') {
        const predictions = payload.data;
        updateDashboard(predictions);
        updateMapMarkers(predictions);
        resetTimer();
    }
};

socket.onclose = () => console.log("WebSocket backend desconectado");

// === Funciones de Actualización de Interfaz ===

function updateDashboard(predictions) {
    const list = document.getElementById('delay-items');
    const count = document.getElementById('alert-count');
    
    list.innerHTML = '';
    count.textContent = predictions.length;

    if(predictions.length === 0) {
        list.innerHTML = '<li class="empty-state">No hay retrasos reportados. Tráfico fluido en el buffer.</li>';
        return;
    }

    predictions.forEach(p => {
        const li = document.createElement('li');
        const isDanger = p.status === 'delayed';
        li.className = `delay-item ${isDanger ? 'danger' : 'warning'}`;
        
        li.innerHTML = `
            <span class="station-id">Parada ID #${p.station_id}</span>
            <span class="delay-time ${isDanger ? 'danger-text' : 'warn-text'}">
                +${p.delay_minutes} min
            </span>
        `;
        list.appendChild(li);
    });
}

function updateMapMarkers(predictions) {
    // Limpiar alertas de todos
    Object.values(markersMap).forEach(marker => {
        const el = marker.getElement();
        if(el) el.classList.remove('delayed');
    });

    // Remarcar las de la predicción que acaba de llegar (simulando los output models)
    predictions.forEach(p => {
        const marker = markersMap[p.station_id];
        if (marker) {
            const el = marker.getElement();
            if(el) {
                // Leaflet no tiene reactividad para añadir clases dinámicas a iconDiv por la cara,
                // Manipulamos el DOM devuelto
                el.classList.add('delayed');
            }
        }
    });
}

// === Temporizador Visual T-Minus ===
let secondsRemaining = 60;
setInterval(() => {
    secondsRemaining--;
    if(secondsRemaining < 0) secondsRemaining = 0;
    document.getElementById('next-update').textContent = `T-${secondsRemaining}s`;
}, 1000);

function resetTimer() {
    secondsRemaining = 60;
}
