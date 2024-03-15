var load_sound = document.getElementById("load-sound");
var click_sound = document.getElementById("click-sound");

var dropZone = document.getElementById("drop_zone");
var fileInput = document.getElementById("image");
var searchInput = document.getElementById("search");
var tokenInput = document.getElementById("token");
var urlInput = document.getElementById("url");
const buttons = document.querySelectorAll(".button");
let scaledImageURL = null;
let segmentationData = null;

dropZone.addEventListener("click", function () {
	fileInput.click();
});
dropZone.addEventListener("dragover", (event) => event.preventDefault());
dropZone.addEventListener("drop", (event) => {
	event.preventDefault();
	handleFileDrop(event.dataTransfer.files[0]);
});
fileInput.addEventListener("change", function () {
	if (this.files && this.files[0]) {
		load_sound.play();
		handleFileDrop(this.files[0]);
		this.value = "";
	}
});
const disableButtonAndShowSpinner = (button) => {
	click_sound.play();
	button.disabled = true;
	const spinner = button.querySelector(".spinner");
	spinner.style.display = "inline-block";
	const icon = button.querySelector(".btn-icon");
	icon.style.display = "none";
};
const enableButtonAndHideSpinner = (button) => {
	button.disabled = false;
	const spinner = button.querySelector(".spinner");
	spinner.style.display = "none";
	const icon = button.querySelector(".btn-icon");
	icon.style.display = "block";
};

buttons.forEach((button) => {
	button.addEventListener("click", () => {
		if (button.tagName.toLowerCase() === "button") {
			if (button.id === "segmentButton") {
				if (searchInput.value.trim()) {
					disableButtonAndShowSpinner(button);
					segmentImage(() => enableButtonAndHideSpinner(button));
				}
			} else if (button.id === "submitButton") {
				if (
					segmentationData &&
					tokenInput.value.trim() &&
					urlInput.value.trim()
				) {
					disableButtonAndShowSpinner(button);
					submitToCloud(() => enableButtonAndHideSpinner(button));
				}
			} else if (button.id === "deleteButton") {
				if (tokenInput.value.trim() && urlInput.value.trim()) {
					disableButtonAndShowSpinner(button);
					deletefromCloud(() => enableButtonAndHideSpinner(button));
				}
			}
		}
	});
});

function handleFileDrop(file) {
	load_sound.play();
	const formData = new FormData();
	formData.append("image", file);
	fetch("/scale", { method: "POST", body: formData })
		.then((response) => response.json())
		.then((data) => {
			const imageUrl = "data:image/jpeg;base64," + data.image;
			scaledImageURL = imageUrl;
			previewImage(imageUrl, data.height, data.width);
			segmentButton.disabled = false;
		})
		.catch((error) => {
			console.error("Error:", error);
		});
}

function previewImage(url, height, width) {
	const mainContainer = document.querySelector(".main-container");

	dropZone.style.backgroundImage = `url(${url})`;
	dropZone.style.backgroundSize = "contain";
	dropZone.style.backgroundRepeat = "no-repeat";
	dropZone.style.backgroundPosition = "center";

	if (mainContainer) {
		const maxWidth = window.getComputedStyle(mainContainer).maxWidth;
		dropZone.style.maxWidth = maxWidth;
	}
	dropZone.maxHeight = height;
	dropZone.innerText = "";
}

function segmentImage(callback) {
	const imageContent = scaledImageURL.split(",")[1];
	const payload = {
		image: imageContent,
		search: searchInput.value,
	};

	fetch("/segment", {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify(payload),
	})
		.then((response) => {
			if (!response.ok) {
				throw new Error(`HTTP error! status: ${response.status}`);
			}
			return response.json();
		})
		.then((data) => {
			segmentationData = data;
			const displayImageURL = `data:image/jpeg;base64,${data.image}`;
			previewImage(displayImageURL, data.height, data.width);
			cloudButton.disabled = false;
			if (callback) callback();
		})
		.catch((error) => {
			console.error("Error during segmentation:", error);
			if (callback) callback();
		});
}

function submitToCloud(callback) {
	const payload = {
		formatted_segments: segmentationData.segments,
		token: tokenInput.value,
		url: urlInput.value,
	};

	fetch("/to_cloud", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(payload),
	})
		.then((response) => {
			if (response.status === 200) {
				return response.json();
			} else {
				throw new Error(
					"Failed to send data to cloud. Status code: " +
						response.status
				);
			}
		})
		.then((data) => {
			alert("Data successfully sent to cloud!");
			if (callback) callback();
		})
		.catch((error) => {
			console.error("Error:", error);
			alert("Failed to send data to cloud.");
			if (callback) callback();
		});
}

function deletefromCloud(callback) {
	const payload = {
		token: tokenInput.value,
		url: urlInput.value,
	};
	fetch("/delete", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(payload),
	})
		.then((response) => {
			if (response.status === 200) {
				return response.json();
			} else {
				throw new Error(
					"Failed to send data to cloud. Status code: " +
						response.status
				);
			}
		})
		.then((data) => {
			alert("Data successfully sent to cloud!");
			if (callback) callback();
		})
		.catch((error) => {
			console.error("Error:", error);
			alert("Failed to send data to cloud.");
			if (callback) callback();
		});
}
