document.addEventListener("DOMContentLoaded", () => {
  const forms = document.querySelectorAll("[data-loading-form]");
  forms.forEach((form) => {
    form.addEventListener("submit", () => {
      const loader = form.querySelector("[data-loader]");
      const submitButton = form.querySelector("[data-submit]");
      if (loader) {
        loader.classList.remove("d-none");
      }
      if (submitButton) {
        submitButton.disabled = true;
        submitButton.innerHTML = '<i class="fa-solid fa-gear fa-spin me-2"></i>Processing...';
      }
    });
  });

  const fileInput = document.getElementById("face_image");
  const previewImage = document.getElementById("uploadPreviewImage");
  const previewText = document.getElementById("uploadPreviewText");

  if (fileInput && previewImage && previewText) {
    fileInput.addEventListener("change", (event) => {
      const target = event.target;
      const file = target.files && target.files[0];
      if (!file) {
        previewImage.classList.add("d-none");
        previewImage.removeAttribute("src");
        previewText.classList.remove("d-none");
        previewText.textContent = "Apercu local de votre image";
        return;
      }

      const reader = new FileReader();
      reader.onload = (readerEvent) => {
        const result = readerEvent.target && readerEvent.target.result;
        if (!result) {
          return;
        }
        previewImage.src = String(result);
        previewImage.classList.remove("d-none");
        previewText.classList.add("d-none");
      };
      reader.readAsDataURL(file);
    });
  }

  const revealElements = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries, instance) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            instance.unobserve(entry.target);
          }
        });
      },
      {
        threshold: 0.15,
      }
    );

    revealElements.forEach((element) => observer.observe(element));
  } else {
    revealElements.forEach((element) => element.classList.add("visible"));
  }
});
