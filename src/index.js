import './styles.css';

document.addEventListener('DOMContentLoaded', () => {
    const predictButton = document.getElementById('predict-button');
    const fileUpload = document.getElementById('file-upload');
    const predictionsDiv = document.getElementById('predictions');
    const errorMessageDiv = document.getElementById('error-message');

    predictButton.addEventListener('click', async () => {
        const file = fileUpload.files[0];

        if (!file) {
            errorMessageDiv.textContent = 'Please upload a CSV file.';
            predictionsDiv.textContent = '';
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        predictionsDiv.textContent = 'Predicting...';
        errorMessageDiv.textContent = '';

        try {
            const response = await fetch(`${process.env.BACKEND_URL}/predict`, { // Use environment variable
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                errorMessageDiv.textContent = `Error: ${errorData.error || response.statusText}`;
                predictionsDiv.textContent = '';
                return;
            }

            const data = await response.json();

            if (data.predictions && Array.isArray(data.predictions)) {
                let predictionsHTML = '';
                data.predictions.forEach((prediction, index) => {
                    predictionsHTML += `<p>Order ${index + 1}: Predicted Delivery Time: ${prediction} minutes</p>`;
                });
                predictionsDiv.innerHTML = predictionsHTML;
            } else if (data.error) {
                errorMessageDiv.textContent = `Error: ${data.error}`;
                predictionsDiv.textContent = '';
            } else {
                errorMessageDiv.textContent = 'Unexpected response from the server.';
                predictionsDiv.textContent = '';
            }

        } catch (error) {
            errorMessageDiv.textContent = `Error during prediction: ${error}`;
            predictionsDiv.textContent = '';
        }
    });
});