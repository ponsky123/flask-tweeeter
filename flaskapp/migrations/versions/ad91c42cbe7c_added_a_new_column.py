"""added a new column

Revision ID: ad91c42cbe7c
Revises: 6db061a61f39
Create Date: 2020-12-06 13:01:22.720457

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ad91c42cbe7c'
down_revision = '6db061a61f39'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('action', sa.Column('label', sa.Integer(), nullable=True))
    
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    
    op.drop_column('action', 'label')
    # ### end Alembic commands ###
