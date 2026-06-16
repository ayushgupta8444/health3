document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadSection = document.getElementById("upload-section");
    const loadingIndicator = document.getElementById("loading-indicator");
    const dashboardSection = document.getElementById("dashboard-section");
    const doshaFilter = document.getElementById("dosha-filter");
    const clearBtn = document.getElementById("clear-btn");

    // Chart instances
    let ageChartInstance = null;
    let genderChartInstance = null;
    let visitChartInstance = null;

    // Default Chart.js Configuration
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.color = '#64748b';

    // Drag and drop handlers
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Upload functionality
    async function handleFileUpload(file) {
        dropZone.classList.add("hidden");
        loadingIndicator.classList.remove("hidden");

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            
            if (res.ok) {
                await loadDashboardData();
                uploadSection.classList.add("hidden");
                dashboardSection.classList.remove("hidden");
                // Animate entry
                dashboardSection.style.opacity = 0;
                setTimeout(() => {
                    dashboardSection.style.transition = "opacity 0.5s ease";
                    dashboardSection.style.opacity = 1;
                }, 100);
            } else {
                alert(data.message || "Error uploading file");
                resetUpload();
            }
        } catch (err) {
            console.error(err);
            alert("Network error occurred.");
            resetUpload();
        }
    }

    function resetUpload() {
        loadingIndicator.classList.add("hidden");
        dropZone.classList.remove("hidden");
        fileInput.value = "";
    }

    // Load Dashboard Data
    async function loadDashboardData(dosha = "All Diseases") {
        try {
            const res = await fetch(`/api/dashboard_data?dosha=${encodeURIComponent(dosha)}`);
            if (!res.ok) return;
            const data = await res.json();

            // Update Metrics
            document.getElementById("metric-patients").textContent = data.metrics.total_patients.toLocaleString();
            document.getElementById("metric-recovery").textContent = data.metrics.avg_recovery;
            document.getElementById("metric-condition").textContent = data.metrics.common_condition;

            // Populate Filter Options only once
            if (dosha === "All Diseases" && doshaFilter.options.length <= 1) {
                doshaFilter.innerHTML = "";
                data.doshas.forEach(d => {
                    const option = document.createElement("option");
                    option.value = d;
                    option.textContent = d;
                    doshaFilter.appendChild(option);
                });
            }

            renderCharts(data.charts);
        } catch (err) {
            console.error("Error loading dashboard data:", err);
        }
    }

    // Render Charts
    function renderCharts(chartsData) {
        // Age Chart
        const ageCtx = document.getElementById('ageChart').getContext('2d');
        if (ageChartInstance) ageChartInstance.destroy();
        ageChartInstance = new Chart(ageCtx, {
            type: 'bar',
            data: {
                labels: Object.keys(chartsData.age_treatment),
                datasets: [{
                    label: 'Avg Treatment Days',
                    data: Object.values(chartsData.age_treatment),
                    backgroundColor: 'rgba(14, 165, 233, 0.8)',
                    borderRadius: 8
                }]
            },
            options: { responsive: true, plugins: { legend: { display: false } } }
        });

        // Gender Chart
        const genderCtx = document.getElementById('genderChart').getContext('2d');
        if (genderChartInstance) genderChartInstance.destroy();
        genderChartInstance = new Chart(genderCtx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(chartsData.gender_dist),
                datasets: [{
                    data: Object.values(chartsData.gender_dist),
                    backgroundColor: ['#ec4899', '#3b82f6', '#8b5cf6'],
                    borderWidth: 0
                }]
            },
            options: { responsive: true, cutout: '70%' }
        });

        // Visit Type Chart
        const visitCtx = document.getElementById('visitChart').getContext('2d');
        if (visitChartInstance) visitChartInstance.destroy();
        visitChartInstance = new Chart(visitCtx, {
            type: 'bar',
            data: {
                labels: Object.keys(chartsData.visit_type),
                datasets: [{
                    label: 'Patient Count',
                    data: Object.values(chartsData.visit_type),
                    backgroundColor: 'rgba(16, 185, 129, 0.8)',
                    borderRadius: 8
                }]
            },
            options: { responsive: true, indexAxis: 'y', plugins: { legend: { display: false } } }
        });
    }

    // Event Listeners for Filters and Controls
    doshaFilter.addEventListener("change", (e) => {
        loadDashboardData(e.target.value);
    });

    clearBtn.addEventListener("click", async () => {
        await fetch("/api/clear", { method: "POST" });
        dashboardSection.classList.add("hidden");
        uploadSection.classList.remove("hidden");
        resetUpload();
        doshaFilter.innerHTML = '<option value="All Diseases">All Diseases</option>';
    });
});
