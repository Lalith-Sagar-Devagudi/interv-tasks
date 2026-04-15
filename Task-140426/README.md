# Project Template Repository

This repository serves as a reusable template for projects, designed to streamline development and maintain consistency across projects. It includes a well-organized structure for backend, frontend, testing, and deployment, ensuring a smooth workflow for your team.

## Features
- **Backend:** Modular and scalable backend setup with support for Python frameworks like FastAPI.
- **Frontend:** React-based frontend structure with TypeScript support.
- **Testing:** Integrated unit and integration testing frameworks for backend and frontend.
- **Deployment:** Docker Compose configuration for orchestrating services.
- **Documentation:** Organized `docs/` directory for comprehensive project documentation.
- **Environment Configuration:** `.env.example` for managing environment variables.
- **CI/CD:** GitHub Actions workflows for automated testing and deployment.

## Repository Structure
```
project-template/
├── .github/                # GitHub-specific configurations
│   ├── ISSUE_TEMPLATE/     # Templates for reporting issues
│   └── PULL_REQUEST_TEMPLATE/  # Templates for pull requests
├── backend/                # Backend services
│   ├── app/                # Application code
│   ├── tests/              # Backend-specific tests
│   └── requirements.txt    # Backend dependencies
├── frontend/               # Frontend application
│   ├── src/                # Source code
│   ├── public/             # Public assets
│   └── package.json        # Frontend dependencies
├── docs/                   # Documentation
├── scripts/                # Deployment and utility scripts
├── .env.example            # Example environment variables
├── .gitignore              # Git ignore file
├── docker-compose.yml      # Docker Compose configuration
├── LICENSE                 # License information
└── README.md               # Project overview
```

## Getting Started

### Prerequisites
- **Backend:** Python 3.8 or higher
- **Frontend:** Angular JS (LTS version)
- **Docker:** Docker and Docker Compose installed on your system

### Setup
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/NeuGenAI-Solutions/project-kun-demo.git
   ```

2. **Configure Environment Variables:**
   - Copy `.env.example` to `.env` and update it with your environment-specific settings.

3. **Install Dependencies:**
     ```bash
     poetry install
     ```

4. **Run Services:**
   - Backend:
     ```bash
     cd backend/app
     poetry run uvicorn app:app --reload
     ```
   - Frontend:
     ```bash
     cd frontend
     python -m http.server 8080
     ```
5. **Access the Application:**
    - Open your browser and navigate to `http://localhost:8080/#!` to access the frontend application.

### Testing
- **Backend Tests:**
  ```bash
  cd backend
  pytest
  ```
- **Frontend Tests:**
  ```bash
  cd frontend
  npm test
  ```

## Contributing
To contribute to this template:
1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request.

## License
This template is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Feedback and Support
If you encounter issues or have suggestions, feel free to create an issue or contribute to the project. Your feedback is invaluable!

