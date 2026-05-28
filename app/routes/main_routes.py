from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def dashboard():
    return render_template('dashboard.html')

@main_bp.route('/editor')
def editor():
    return render_template('editor.html', title="Model Editor")

@main_bp.route('/train')
def training():
    return render_template('training.html', title="Model Training")

@main_bp.route('/projects')
def projects():
    return render_template('projects.html', title="Projects")

@main_bp.route('/playground')
def playground():
    return render_template('playground.html', title="Playground")

@main_bp.route('/run/<run_id>')
def run_details(run_id):
    return render_template('run_details.html', run_id=run_id, title="Training Run Details")
