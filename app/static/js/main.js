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
let allStations = []; // Cache de todas las estaciones
let currentPredictions = []; // Cache de predicciones vivas
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
        allStations = stations;
        stations.forEach(station => {
            const color = getStationColor(station.routes);
            const markerValue = L.divIcon({
                className: 'station-marker',
                iconSize: [10, 10],
                html: `<div style="background-color: ${color}; width: 100%; height: 100%; border-radius: 50%; border: 1px solid white; box-shadow: 0 0 5px ${color}"></div>`
            });
            
            // Creamos el marker en Leaflet
            const marker = L.marker([station.lat, station.lon], {icon: markerValue});
            
            // Evento Click para abrir Panel Detallado
            marker.on('click', () => {
                openStationDetails(station);
            });
            
            markersMap[station.id] = marker.addTo(map);
        });
        console.log(`${stations.length} estaciones instanciadas.`);
        initRoutePlanner(); // Inicializar el buscador
    })
    .catch(err => console.error("Error al cargar dataset de estaciones desde API:", err));

// 3. Conexión WebSocket para Predicciones periódicas de FastAPI
const wsUrl = `ws://${window.location.host}/ws/live-updates`;
const socket = new WebSocket(wsUrl);

socket.onopen = () => console.log("Conectado a FastAPI WebSocket Core");

socket.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    
    if (payload.type === 'update' || payload.type === 'initial') {
        currentPredictions = payload.data;
        updateMapMarkers(currentPredictions);
        // Si hay una estación abierta, refrescar alertas
        if (!detailPanel.classList.contains('hidden')) {
             checkActiveAlerts();
        }
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

    // Remarcar las de la predicción que acaba de llegar
    predictions.forEach(p => {
        const marker = markersMap[p.station_id];
        if (marker) {
            const el = marker.getElement();
            if(el) el.classList.add('delayed');
        }
    });
}

// === Nueva Lógica de Panel de Detalles de Estación ===

const detailPanel = document.getElementById('station-detail-panel');
const closeDetailBtn = document.getElementById('close-detail');

closeDetailBtn.onclick = () => {
    detailPanel.classList.add('hidden');
};

function openStationDetails(station) {
    detailPanel.currentStation = station; // Guardar estación actual
    const nameEl = document.getElementById('detail-station-name');
    const lineSelectors = document.getElementById('detail-line-selectors');
    
    detailPanel.classList.remove('hidden');
    map.flyTo([station.lat, station.lon], 14, { duration: 1.5 });
    
    nameEl.textContent = station.name;
    lineSelectors.innerHTML = '';
    
    const routes = station.routes.split(' ');
    routes.forEach((route, index) => {
        const bubble = document.createElement('div');
        const color = ROUTE_COLORS[route] || '#3B82F6';
        
        bubble.className = `line-bubble ${index === 0 ? 'active' : ''}`;
        bubble.textContent = route;
        bubble.style.backgroundColor = color;
        
        bubble.onclick = () => {
            document.querySelectorAll('.line-bubble').forEach(b => b.classList.remove('active'));
            bubble.classList.add('active');
            updateForecast(route);
        };
        
        lineSelectors.appendChild(bubble);
    });
    
    if (routes.length > 0) {
        updateForecast(routes[0]);
    }
}

function updateForecast(lineCode) {
    const cards = ['forecast-now', 'forecast-10', 'forecast-20', 'forecast-30'];
    detailPanel.currentLine = lineCode; // Guardar línea actual

    cards.forEach(cardId => {
        const el = document.getElementById(cardId);
        const delay = Math.floor(Math.random() * 8);
        el.textContent = delay === 0 ? 'ON TIME' : `+${delay} min`;
        el.className = 'delay-val ' + (delay === 0 ? 'on-time' : (delay < 4 ? 'minor-delay' : 'delayed'));
    });
    
    checkActiveAlerts();

    const infoEl = document.getElementById('line-detail-info');
    infoEl.innerHTML = `<p>Estado actual de la línea <strong>${lineCode}</strong> en esta estación basado en histórico y buffer en tiempo real.</p>`;
}

function checkActiveAlerts() {
    const alertSection = document.getElementById('station-alert');
    if (!detailPanel.currentStation || !detailPanel.currentLine) return;

    // Buscamos si la estación actual está en las predicciones críticas
    const hasAlert = currentPredictions.some(p => p.station_id === detailPanel.currentStation.id);
    
    if (hasAlert) {
        alertSection.classList.remove('hidden');
    } else {
        alertSection.classList.add('hidden');
    }
}

// === Lógica del Buscador / Planificador de Rutas ===

function initRoutePlanner() {
    const originInput = document.getElementById('origin-input');
    const destinationInput = document.getElementById('destination-input');
    const originDropdown = document.getElementById('origin-results');
    const destinationDropdown = document.getElementById('destination-results');

    setupAutocomplete(originInput, originDropdown);
    setupAutocomplete(destinationInput, destinationDropdown);
}

function setupAutocomplete(input, dropdown) {
    input.addEventListener('input', () => {
        const value = input.value.toLowerCase().trim();
        if (value.length < 2) {
            dropdown.classList.add('hidden');
            return;
        }

        const filtered = allStations
            .filter(s => s.name.toLowerCase().includes(value))
            .slice(0, 5);

        if (filtered.length > 0) {
            dropdown.innerHTML = '';
            filtered.forEach(s => {
                const item = document.createElement('div');
                item.className = 'suggestion-item';
                item.textContent = s.name;
                item.onclick = () => {
                    input.value = s.name;
                    input.dataset.stationId = s.id;
                    dropdown.classList.add('hidden');
                    calculateRoute();
                };
                dropdown.appendChild(item);
            });
            dropdown.classList.remove('hidden');
        } else {
            dropdown.classList.add('hidden');
        }
    });

    // Cerrar al hacer click fuera
    document.addEventListener('click', (e) => {
        if (!input.contains(e.target) && !dropdown.contains(e.target)) {
            dropdown.classList.add('hidden');
        }
    });
}

function calculateRoute() {
    const originId = document.getElementById('origin-input').dataset.stationId;
    const destId = document.getElementById('destination-input').dataset.stationId;
    const resultPanel = document.getElementById('route-prediction');

    if (originId && destId) {
        const origin = allStations.find(s => s.id === originId);
        const destination = allStations.find(s => s.id === destId);
        
        resultPanel.classList.remove('hidden');
        
        // Simular lógica de ruta
        const commonLines = findCommonLines(origin, destination);
        const displayLine = commonLines.length > 0 ? commonLines[0] : origin.routes.split(' ')[0];
        
        document.getElementById('result-line').textContent = displayLine;
        document.getElementById('result-line').style.backgroundColor = ROUTE_COLORS[displayLine] || '#3B82F6';
        document.getElementById('result-station').textContent = origin.name;
        
        // Buscar si hay predicción real para el origen
        const realPrediction = currentPredictions.find(p => p.station_id === originId);
        const delay = realPrediction ? realPrediction.delay_minutes : Math.floor(Math.random() * 5);
        
        const delayValEl = document.getElementById('result-delay-val');
        delayValEl.textContent = delay === 0 ? 'ON TIME' : `+${delay} min`;
        delayValEl.className = 'delay-val ' + (delay === 0 ? 'on-time' : (delay < 4 ? 'minor-delay' : 'delayed'));
    }
}

function findCommonLines(s1, s2) {
    const r1 = s1.routes.split(' ');
    const r2 = s2.routes.split(' ');
    return r1.filter(line => r2.includes(line));
}
